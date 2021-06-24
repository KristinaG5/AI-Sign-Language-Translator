import sys
import cv2
import os
from sys import platform
import argparse
import json


def extractImages(folder, folder_path, video_files):
    # Converts each video clip into frames
    count = 0
    for video in video_files:
        vidcap = cv2.VideoCapture(f"{folder_path}/{video}")
        success, image = vidcap.read()
        while success:
            cv2.imwrite(f"../../Dataset/data/{folder}/frame%d.jpg" % count, image)
            success, image = vidcap.read()
            count += 1


if __name__ == "__main__":
    """
    Takes a folder of videos and converts each video to frames
    """

    # Flags
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-f", "--folder_path", help="Path to folder of videos to be converted", type=str, required=True)
    parser.add_argument("-s", "--folder", help="Name of the folder to save the frames to", type=str, required=True)
    parser.add_argument("-p", "--openpose_dir", help="Path to the openpose build dir", type=str, required=True)
    parser.add_argument("-m", "--openpose_models_dir", help="Path to the openpose models dir", type=str, required=True)
    args = parser.parse_args()

    try:
        sys.path.append(args.openpose_dir)
        from openpose import pyopenpose as op
    except ImportError as e:
        print("Error: OpenPose library could not be found.")
        raise e

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = args.openpose_models_dir
    params["face"] = True
    params["hand"] = True

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process frames from multiple videos
    video_files = os.listdir(args.folder_path)
    data = []

    if video_files:
        extractImages(args.folder, args.folder_path, video_files)

    sys.exit(-1)
