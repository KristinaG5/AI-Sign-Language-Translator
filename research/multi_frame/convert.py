import os
import json
from sklearn.model_selection import train_test_split


def load_json(path):
    with open(path) as json_file:
        videos = json.load(json_file)
    return videos


def save_to_file(path, data):
    with open(path, "w") as f:
        for item in data:
            if isinstance(item, list):
                for value in item:
                    f.write(str(value) + " ")
            else:
                f.write(str(item))
            f.write("\n")


dataset_folder = "data/frankmocap/ratio/"
jsons = os.listdir(dataset_folder)
data = {}
for file_name in jsons:
    data[file_name[:-5]] = load_json(os.path.join(dataset_folder, file_name))

X_train = []
y_train = []
X_test = []
y_test = []
labels = {"car": 1, "i": 2, "mechanic": 3, "my": 4, "story": 5, "true": 6}

for class_name, videos in data.items():
    for video in videos:
        frames = video["normalised"]
        num_frames = len(frames)
        train_size = int(num_frames * 0.8)

        for frame in frames[:train_size]:
            X_train.append(frame)
            y_train.append(labels[class_name])

        for frame in frames[train_size:]:
            X_test.append(frame)
            y_test.append(labels[class_name])


print(len(X_train), len(X_test))

save_to_file("data/frankmocap/X_train.txt", X_train)
save_to_file("data/frankmocap/X_test.txt", X_test)
save_to_file("data/frankmocap/Y_train.txt", y_train)
save_to_file("data/frankmocap/Y_test.txt", y_test)
