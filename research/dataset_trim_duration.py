import os
import shutil
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


def video_info(input_path):
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

durations = []
for folder in tqdm(os.listdir(dataset_path)):
    folder_path = os.path.join(dataset_path, folder)
    files = os.listdir(folder_path)
    num_files = len(files)

    for file_name in files:
        duration, fps = video_info(os.path.join(folder_path, file_name))
        if duration and fps:
            durations.append(int(duration))

print(max(durations))
plt.hist(durations, bins=max(durations))
plt.show()
