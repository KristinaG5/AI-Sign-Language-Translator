# Copyright (c) Facebook, Inc. and its affiliates.
import sys
import os

directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(directory, "frankmocap"))
sys.path.append(os.path.join(directory, "frankmocap", "detectors", "body_pose_estimator"))
sys.path.append(os.path.join(directory, "frankmocap", "bodymocap"))
sys.path.append(os.path.join(directory, "frankmocap", "extra_data"))

import torch
import numpy as np
import json
import cv2
import mocap_utils.general_utils as gnu
import os.path as osp

from bodymocap.body_mocap_api import BodyMocap
from handmocap.hand_mocap_api import HandMocap
from handmocap.hand_bbox_detector import HandBboxDetector
from integration.copy_and_paste import integration_copy_paste


def __filter_bbox_list(body_bbox_list, hand_bbox_list, single_person):
    # (to make the order as consistent as possible without tracking)
    bbox_size = [(x[2] * x[3]) for x in body_bbox_list]
    idx_big2small = np.argsort(bbox_size)[::-1]
    body_bbox_list = [body_bbox_list[i] for i in idx_big2small]
    hand_bbox_list = [hand_bbox_list[i] for i in idx_big2small]

    if single_person and len(body_bbox_list) > 0:
        body_bbox_list = [
            body_bbox_list[0],
        ]
        hand_bbox_list = [
            hand_bbox_list[0],
        ]

    return body_bbox_list, hand_bbox_list


def run_regress(args, img_original_bgr, body_bbox_list, hand_bbox_list, bbox_detector, body_mocap, hand_mocap):
    cond1 = len(body_bbox_list) > 0 and len(hand_bbox_list) > 0
    cond2 = not args.frankmocap_fast_mode

    # use pre-computed bbox or use slow detection mode
    if cond1 or cond2:
        if not cond1 and cond2:
            # run detection only when bbox is not available
            body_pose_list, body_bbox_list, hand_bbox_list, _ = bbox_detector.detect_hand_bbox(img_original_bgr.copy())
        else:
            print("Use pre-computed bounding boxes")
        assert len(body_bbox_list) == len(hand_bbox_list)

        if len(body_bbox_list) < 1:
            return list(), list(), list()

        # sort the bbox using bbox size
        # only keep on bbox if args.single_person is set
        body_bbox_list, hand_bbox_list = __filter_bbox_list(body_bbox_list, hand_bbox_list, args.single_person)

        # hand & body pose regression
        pred_hand_list = hand_mocap.regress(img_original_bgr, hand_bbox_list, add_margin=True)
        pred_body_list = body_mocap.regress(img_original_bgr, body_bbox_list)
        assert len(hand_bbox_list) == len(pred_hand_list)
        assert len(pred_hand_list) == len(pred_body_list)

    else:
        _, body_bbox_list = bbox_detector.detect_body_bbox(img_original_bgr.copy())

        if len(body_bbox_list) < 1:
            return list(), list(), list()

        # sort the bbox using bbox size
        # only keep on bbox if args.single_person is set
        hand_bbox_list = [
            None,
        ] * len(body_bbox_list)
        body_bbox_list, _ = __filter_bbox_list(body_bbox_list, hand_bbox_list, args.single_person)

        # body regression first
        pred_body_list = body_mocap.regress(img_original_bgr, body_bbox_list)
        assert len(body_bbox_list) == len(pred_body_list)

        # get hand bbox from body
        hand_bbox_list = body_mocap.get_hand_bboxes(pred_body_list, img_original_bgr.shape[:2])
        assert len(pred_body_list) == len(hand_bbox_list)

        # hand regression
        pred_hand_list = hand_mocap.regress(img_original_bgr, hand_bbox_list, add_margin=True)
        assert len(hand_bbox_list) == len(pred_hand_list)

    # integration by copy-and-paste
    integral_output_list = integration_copy_paste(
        pred_body_list, pred_hand_list, body_mocap.smpl, img_original_bgr.shape
    )

    return body_bbox_list, hand_bbox_list, integral_output_list


class Options:
    def __init__(self, input_path):
        self.input_path = input_path
        self.out_dir = "out"
        self.use_smplx = True
        self.frankmocap_fast_mode = False
        self.single_person = False


def estimate_pose(body_data, left_hand_data, right_hand_data):
    """Extract body, left and right hand pose estimation points

    Arguments
    ---------
    body_data : array
        Estimated body key points

    left_hand_data : array
        Estimated left hand key points

    right_hand_data : array
        Estimated right hand key points

    Returns
    -------
    Body, left and right hand key points
    """
    body_frame_points = []
    left_hand_frame_points = []
    right_hand_frame_points = []

    for i in range(0, 8):
        body_frame_points.append(float(body_data[i][0]))
        body_frame_points.append(float(body_data[i][1]))

    for i in range(0, 21):
        right_hand_frame_points.append(float(left_hand_data[i][0]))
        right_hand_frame_points.append(float(left_hand_data[i][1]))
        left_hand_frame_points.append(float(right_hand_data[i][0]))
        left_hand_frame_points.append(float(right_hand_data[i][1]))

    return body_frame_points, right_hand_frame_points, left_hand_frame_points


def run_frank_mocap(input_path, bbox_detector, body_mocap, hand_mocap, fps):
    """Processes video through OpenCV to apply pose estimation

    Arguments
    ---------
    input_path : string
        Path to individual video

    bbox_detector : object
        FrankMocap bounding box detector

    body_mocap : object
        FrankMocap body classifier

    hand_mocap : object
        FrankMocap hand classifier

    fps: int, default 25
        Rate at which to sample video

    Returns
    -------
    Dictionary
    """
    args = Options(input_path)
    input_type = "video"

    device = torch.device("cuda")
    # Setup input data to handle different types of inputs
    cap = cv2.VideoCapture(input_path)
    assert cap.isOpened(), f"Failed in opening video: {input_path}"
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    assert frame_rate <= 60, f"{input_path} has an invalid frame rate of {frame_rate}"
    frame_sampling = round(frame_rate / fps)
    item = {
        "video": input_path,
        "width": cap.get(3),
        "height": cap.get(4),
        "body": [],
        "left_hand": [],
        "right_hand": [],
    }

    cur_frame = 0
    video_frame = 0
    while True:
        # load data
        _, img_original_bgr = cap.read()

        if video_frame < cur_frame:
            video_frame += 1
            continue

        if img_original_bgr is not None:
            video_frame += 1

        cur_frame += 1
        if img_original_bgr is None:
            break

        if cur_frame % frame_sampling == 0:

            # bbox detection
            body_bbox_list, hand_bbox_list = list(), list()

            # regression (includes integration)
            body_bbox_list, hand_bbox_list, pred_output_list = run_regress(
                args, img_original_bgr, body_bbox_list, hand_bbox_list, bbox_detector, body_mocap, hand_mocap
            )

            body_data = pred_output_list[0]["pred_body_joints_img"]
            left_hand_data = pred_output_list[0]["pred_lhand_joints_img"]
            right_hand_data = pred_output_list[0]["pred_rhand_joints_img"]

            body_frame_points, right_hand_frame_points, left_hand_frame_points = estimate_pose(
                body_data, left_hand_data, right_hand_data
            )
            item["body"].append(body_frame_points)
            item["right_hand"].append(right_hand_frame_points)
            item["left_hand"].append(left_hand_frame_points)

            if len(body_bbox_list) < 1:
                print(f"No body deteced: {image_path}")
                continue

    return item


def load_frankmocap(weights_folder):
    """Load FrankMocap

    Arguments
    ---------
    weights_folder : string
        Path to weights

    Returns
    -------
    hand_bbox_detector, body_mocap, hand_mocap
    """
    device = torch.device("cuda")
    hand_bbox_detector = HandBboxDetector("third_view", weights_folder)
    body_mocap = BodyMocap(weights_folder, device=device)
    hand_mocap = HandMocap(weights_folder)

    return hand_bbox_detector, body_mocap, hand_mocap
