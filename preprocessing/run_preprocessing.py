import os
from os.path import dirname, abspath
import argparse
import json
import torch
from tqdm import tqdm
import sys

sys.path.append(dirname(dirname(abspath(__file__))))

from preprocessing.openpose_pose_estimation import load_openpose, run_openpose
from preprocessing.frankmocap_pose_estimation import load_frankmocap, run_frank_mocap
from preprocessing.ratio import get_box_size, get_coordinate_ratio, get_ratio
from preprocessing.angle import calculate_joint_angles
from preprocessing.arguments import get_arguments

op = None
opWrapper = None
hand_bbox_detector = None
body_mocap = None
hand_mocap = None
fps = None


def normalise_video(video, normalisation_type):
    """Returns the normalised pose data depending on the normalisation type

    Arguments
    ---------
    video : dictionary
        Extracted pose information from video

    normalisation_type : string
        Normalisation type to use to normalise the data

    Returns
    -------
    Normalised frame points
    """
    if normalisation_type == "ratio":
        return get_ratio(video)
    else:
        return calculate_joint_angles(video)


def extract_video(
    video_path,
    method,
    normalisation_type,
    op=None,
    opWrapper=None,
    hand_bbox_detector=None,
    body_mocap=None,
    hand_mocap=None,
    fps=25,
):
    """Extracts pose information from video using pose estimation library

    Arguments
    ---------
    video_path : string
        Path to video location to be extracted

    method : string
        Which pose estimation library to use

    normalisation_type : string
        Normalisation type to use to normalise the data

    op, opWrapper : object
        OpenPose classifiers

    hand_bbox_detector, body_mocap, hand_mocap : objects
        FrankMocap classifiers

    fps: int, default 25
        Rate at which to sample video

    Returns
    -------
    Dictionary with normalised frame points
    """
    if method == "openpose":
        video = run_openpose(op, opWrapper, video_path, fps)
    else:
        video = run_frank_mocap(video_path, hand_bbox_detector, body_mocap, hand_mocap, fps)

    assert video
    frames = normalise_video(video, normalisation_type)
    video["normalised"] = frames
    return video


def extract_videos(method, folder_path, normalisation_type):
    """Extracts pose information for multiple videos

    Arguments
    ---------
    method : string
        Which pose estimation library to use

    folder_path : string
        Path to videos folder

    normalisation_type : string
        Normalisation type to use to normalise the data

    Returns
    -------
    Array of extracted pose data
    """
    global op, opWrapper, hand_bbox_detector, body_mocap, hand_mocap, fps

    video_files = os.listdir(folder_path)
    pose_data = []

    for video in video_files:
        video_path = os.path.join(folder_path, video)
        try:
            item = extract_video(
                video_path, method, normalisation_type, op, opWrapper, hand_bbox_detector, body_mocap, hand_mocap, fps
            )
            pose_data.append(item)
        except Exception:
            print(f"Failed to extract pose from {video_path}")

    return pose_data


def preprocess(folder_path, output_directory, method, normalisation_type):
    """Applies preprocessing steps and saves the generated dataset

    Arguments
    ---------
    folder_path : string
        Path to videos folder

    output_directory : string
        Path to save the JSON files created

    method : string
        Which pose estimation library to use

    normalisation_type : string
        Normalisation type to use to normalise the data
    """
    os.makedirs(output_directory, exist_ok=True)
    class_folders = os.listdir(folder_path)

    for class_folder in tqdm(class_folders):
        path_to_video_folder = os.path.join(folder_path, class_folder)
        path_to_save_json = os.path.join(output_directory, class_folder.split(".")[0] + ".json")
        if not os.path.isfile(path_to_save_json):
            normalised = extract_videos(method, path_to_video_folder, normalisation_type)

            with open(path_to_save_json, "w") as outfile:
                json_data = json.dumps(normalised, indent=True)
                outfile.write(json_data)
                outfile.write("\n")
        else:
            print(class_folder, "already exists")


if __name__ == "__main__":
    """
    Script to apply pose estimation and normalisation to generate dataset
    """
    args = get_arguments()
    fps = args.fps

    if args.method == "openpose":
        op, opWrapper = load_openpose(args.openpose_dir, args.openpose_models_dir)
    else:
        hand_bbox_detector, body_mocap, hand_mocap = load_frankmocap(args.frankmocap_dir)

    preprocess(args.folder_path, args.output_dir, args.method, args.normalisation_type)
