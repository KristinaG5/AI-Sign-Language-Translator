import json
import pickle
import numpy as np
import os
from load_multi_frame import flatten_videos, pad_video

WINDOW_SIZE = 1500

with open("../../../data/validation_ratio.json") as f:
    videos = json.load(f)

X = [video["normalised"] for video in videos]
X = flatten_videos(X)
windows = []
num_samples = []
video_labels = {
    "23640": [
        {"word": "my", "start": 0, "end": 1},
        {"word": "car", "start": 1, "end": 2},
        {"word": "mechanic", "start": 2, "end": 3},
    ],
    "9575": [
        {"word": "i", "start": 0, "end": 2},
        {"word": "true", "start": 2, "end": 5},
        {"word": "story", "start": 5, "end": 8},
    ],
}
video_names = list(video_labels.keys())

for sample in X:
    counter = 0
    for i in range(WINDOW_SIZE, len(sample), 100):
        windows.append(sample[i - WINDOW_SIZE : i])
        counter += 1

    end_window = sample[i:]
    end_window = pad_video(end_window, WINDOW_SIZE)
    windows.append(end_window)
    num_samples.append(counter + 1)


def calculate_score(predicitions):
    offset = 0
    averages = []

    for i in range(len(video_names)):
        labels = video_labels[video_names[i]]
        scores = []

        for label in labels:
            section_pred = predicitions[label["start"] + offset : label["end"] + offset]
            score = 0
            for item in section_pred:
                if item == label["word"]:
                    score += 1
            score = score / len(section_pred)
            scores.append(score)

        average = sum(scores) / len(scores)
        print(video_names[i], scores, average)
        averages.append(average)
        offset += num_samples[i]

    print(sum(averages) / len(averages))


X = [np.array(l) for l in windows]
model_folder = "multi_frankmocap_overlapping_15/"
for classifier in os.listdir(model_folder):
    clf = pickle.load(open(os.path.join(model_folder, classifier), "rb"))
    pred = clf.predict(X)
    print(classifier)
    print(pred)
    calculate_score(pred)
    # counter = 0
    # for i in range(len(num_samples)):
    #     size = num_samples[i]
    #     print(video_names[i], pred[counter:size+counter])
    #     counter += size
    print()
