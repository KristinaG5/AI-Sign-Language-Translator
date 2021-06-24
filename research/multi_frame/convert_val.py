import os
import json
from sklearn.model_selection import train_test_split


def save_to_file(path, data):
    with open(path, "w") as f:
        for item in data:
            if isinstance(item, list):
                for value in item:
                    f.write(str(value) + " ")
            else:
                f.write(str(item))
            f.write("\n")


dataset_folder = "data/validation/70148/BF3n_70148.json"
with open(dataset_folder) as json_file:
    video = json.load(json_file)

X_train = []
y_train = []

for v in video:
    frames = v["normalised"]
    num_frames = len(frames)
    for frame in frames:
        X_train.append(frame)


save_to_file("data/validation/70148/X_train.txt", X_train)
