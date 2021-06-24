# Copyright (c) Facebook, Inc. and its affiliates.

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import os
import os.path as osp
import sys
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from bodymocap.body_bbox_detector import BodyPoseEstimator
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


class Third_View_Detector(BodyPoseEstimator):
    """
    Hand Detector for third-view input.
    It combines a body pose estimator (https://github.com/jhugestar/lightweight-human-pose-estimation.pytorch.git)
    with a type-agnostic hand detector (https://github.com/ddshan/hand_detector.d2)
    """

    def __init__(self, config, weights, pose2d_checkpoint):
        super(Third_View_Detector, self).__init__(pose2d_checkpoint)
        print("Loading Third View Hand Detector")
        self.__load_hand_detector(config, weights)

    def __load_hand_detector(self, config, weights):
        # load cfg and model
        cfg = get_cfg()
        cfg.merge_from_file(config)
        cfg.MODEL.WEIGHTS = weights
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # 0.3 , use low thresh to increase recall
        self.hand_detector = DefaultPredictor(cfg)

    def __get_raw_hand_bbox(self, img):
        bbox_tensor = self.hand_detector(img)["instances"].pred_boxes
        bboxes = bbox_tensor.tensor.cpu().numpy()
        return bboxes

    def detect_hand_bbox(self, img):
        """
        output:
            body_bbox: [min_x, min_y, width, height]
            hand_bbox: [x0, y0, x1, y1]
        Note:
            len(body_bbox) == len(hand_bbox), where hand_bbox can be None if not valid
        """
        # get body pose
        body_pose_list, body_bbox_list = self.detect_body_pose(img)
        # assert len(body_pose_list) == 1, "Current version only supports one person"

        # get raw hand bboxes
        raw_hand_bboxes = self.__get_raw_hand_bbox(img)
        hand_bbox_list = [
            None,
        ] * len(body_pose_list)
        num_bbox = raw_hand_bboxes.shape[0]

        if num_bbox > 0:
            for idx, body_pose in enumerate(body_pose_list):
                # By default, we use distance to ankle to distinguish left/right,
                # if ankle is unavailable, use elbow, then use shoulder.
                # The joints used by two arms should exactly the same)
                dist_left_arm = np.ones((num_bbox,)) * float("inf")
                dist_right_arm = np.ones((num_bbox,)) * float("inf")
                hand_bboxes = dict(left_hand=None, right_hand=None)
                # left arm
                if body_pose[7][0] > 0 and body_pose[6][0] > 0:
                    # distance between elbow and ankle
                    dist_wrist_elbow = np.linalg.norm(body_pose[7] - body_pose[6])
                    for i in range(num_bbox):
                        bbox = raw_hand_bboxes[i]
                        c_x = (bbox[0] + bbox[2]) / 2
                        c_y = (bbox[1] + bbox[3]) / 2
                        center = np.array([c_x, c_y])
                        dist_bbox_ankle = np.linalg.norm(center - body_pose[7])
                        if dist_bbox_ankle < dist_wrist_elbow * 1.5:
                            dist_left_arm[i] = np.linalg.norm(center - body_pose[7])
                # right arm
                if body_pose[4][0] > 0 and body_pose[3][0] > 0:
                    # distance between elbow and ankle
                    dist_wrist_elbow = np.linalg.norm(body_pose[3] - body_pose[4])
                    for i in range(num_bbox):
                        bbox = raw_hand_bboxes[i]
                        c_x = (bbox[0] + bbox[2]) / 2
                        c_y = (bbox[1] + bbox[3]) / 2
                        center = np.array([c_x, c_y])
                        dist_bbox_ankle = np.linalg.norm(center - body_pose[4])
                        if dist_bbox_ankle < dist_wrist_elbow * 1.5:
                            dist_right_arm[i] = np.linalg.norm(center - body_pose[4])

                # assign bboxes
                # hand_bboxes = dict()
                left_id = np.argmin(dist_left_arm)
                right_id = np.argmin(dist_right_arm)

                if dist_left_arm[left_id] < float("inf"):
                    hand_bboxes["left_hand"] = raw_hand_bboxes[left_id].copy()
                if dist_right_arm[right_id] < float("inf"):
                    hand_bboxes["right_hand"] = raw_hand_bboxes[right_id].copy()

                hand_bbox_list[idx] = hand_bboxes

        assert len(body_bbox_list) == len(hand_bbox_list)
        return body_pose_list, body_bbox_list, hand_bbox_list, raw_hand_bboxes


class HandBboxDetector(object):
    def __init__(self, view_type, weights_folder):
        """
        args:
            view_type: third_view or ego_centric.
        """
        self.view_type = view_type
        config = os.path.join(weights_folder, "hand_module/faster_rcnn_X_101_32x8d_FPN_3x_100DOH.yaml")
        weights = os.path.join(weights_folder, "hand_module/hand_detector/model_0529999.pth")
        pose2d_checkpoint = os.path.join(weights_folder, "body_module/body_pose_estimator/checkpoint_iter_370000.pth")

        if view_type == "third_view":
            self.model = Third_View_Detector(config, weights, pose2d_checkpoint)
        else:
            print("Invalid view_type")
            assert False

    def detect_body_bbox(self, img_bgr):
        return self.model.detect_body_pose(img_bgr)

    def detect_hand_bbox(self, img_bgr):
        """
        args:
            img_bgr: Raw image with BGR order (cv2 default). Currently assumes BGR
        output:
            body_pose_list: body poses
            bbox_bbox_list: list of bboxes. Each bbox has XHWH form (min_x, min_y, width, height)
            hand_bbox_list: each element is
            dict(
                left_hand = None / [min_x, min_y, width, height]
                right_hand = None / [min_x, min_y, width, height]
            )
            raw_hand_bboxes: list of raw hand detection, each element is [min_x, min_y, width, height]
        """
        output = self.model.detect_hand_bbox(img_bgr)
        body_pose_list, body_bbox_list, hand_bbox_list, raw_hand_bboxes = output

        # convert raw_hand_bboxes from (x0, y0, x1, y1) to (x0, y0, w, h)
        if raw_hand_bboxes is not None:
            for i in range(raw_hand_bboxes.shape[0]):
                bbox = raw_hand_bboxes[i]
                x0, y0, x1, y1 = bbox
                raw_hand_bboxes[i] = np.array([x0, y0, x1 - x0, y1 - y0])

        # convert hand_bbox_list from (x0, y0, x1, y1) to (x0, y0, w, h)
        for hand_bbox in hand_bbox_list:
            if hand_bbox is not None:
                for hand_type in hand_bbox:
                    bbox = hand_bbox[hand_type]
                    if bbox is not None:
                        x0, y0, x1, y1 = bbox
                        hand_bbox[hand_type] = np.array([x0, y0, x1 - x0, y1 - y0])

        return body_pose_list, body_bbox_list, hand_bbox_list, raw_hand_bboxes
