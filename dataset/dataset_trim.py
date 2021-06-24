import os
import shutil
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


def video_info(input_path):
    """Uses OpenCV to extract video information

    Arguments
    ---------
    input_path : string
    Path to each video

    Returns
    ---------
    Duration and FPS values
    """
    try:
        cap = cv2.VideoCapture(input_path)
        assert cap.isOpened(), f"Failed in opening video: {input_path}"

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        return duration, fps
    except AssertionError:
        return None, None


dataset_path = os.path.join("data", "full_dataset")
destination_path = os.path.join("data", "trimmed_dataset_5_val")
os.makedirs(destination_path)


def class_name_is_valid(class_name):
    """Check if the filename is valid

    Arguments
    ---------
    class_name : string
    Name of the file

    Returns
    ---------
    Boolean
    """
    if len(class_name) == 1 and class_name not in ["a", "i"]:
        return False
    if "-" in class_name:
        return False
    return True


def video_is_valid(duration, fps):
    """Checks if video is valid

    Arguments
    ---------
    duration : int
    Length of video

    fps : int
    FPS of video

    Returns
    ---------
    Boolean
    """
    if not duration or not fps:
        return False
    if int(duration) > 3 or fps > 30:
        return False
    return True


"""Cleans the word dataset

Returns
---------
A folder containing the cleaned dataset
"""
num_videos_per_class = 5
for folder in tqdm(os.listdir(dataset_path)):
    folder_path = os.path.join(dataset_path, folder)
    files = os.listdir(folder_path)
    num_files = len(files)

    if num_files >= num_videos_per_class and class_name_is_valid(folder):
        class_folder = os.path.join(destination_path, folder)
        shutil.copytree(folder_path, class_folder)

        for file_name in files:
            video_path = os.path.join(class_folder, file_name)
            duration, fps = video_info(video_path)
            if not video_is_valid(duration, fps):
                os.remove(video_path)

        files = os.listdir(class_folder)
        num_files = len(files)

        if num_files > num_videos_per_class:
            files_to_remove = files[num_videos_per_class:]
            for file_name in files_to_remove:
                os.remove(os.path.join(class_folder, file_name))
        elif num_files < num_videos_per_class:
            print(f"deleted {class_folder}")
            shutil.rmtree(class_folder)
