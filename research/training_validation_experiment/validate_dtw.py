import json
import pickle
import numpy as np
import os
import sys
from os.path import dirname, abspath

sys.path.append(dirname(dirname(dirname(abspath(__file__)))))
import training.train

with open("data/validation_ratio.json") as f:
    videos = json.load(f)

video_labels = {
    "23640": [
        {"word": "my", "start": 0, "end": 5},
        {"word": "car", "start": 6, "end": 10},
        {"word": "mechanic", "start": 11, "end": 26},
    ],
    "9575": [
        {"word": "i", "start": 0, "end": 23},
        {"word": "true", "start": 26, "end": 54},
        {"word": "story", "start": 55, "end": 76},
    ],
}
video_names = list(video_labels.keys())


def calculate_score(predicitions):
    offset = 0
    averages = []

    for i in range(len(video_names)):
        labels = video_labels[video_names[i]]
        scores = []

        for label in labels:
            pred = predicitions[i][label["start"] : label["end"]]
            score = 0
            for item in pred:
                if item == label["word"]:
                    score += 1
            score = score / len(pred)
            scores.append(score)

        average = sum(scores) / len(scores)
        print(video_names[i], scores, average)
        averages.append(average)

    print(sum(averages) / len(averages))


clf = pickle.load(open("standard_25fps_small.pickle", "rb"))
clf.max_warping_window = 10
clf.n_neighbors = 1
preds = []
for video in videos:
    confidences = clf.predict(np.array(video["normalised"]))
    predictions = np.argmax(confidences, axis=1)

    words = []
    for l in predictions:
        words.append(clf.classes[l])
    preds.append(words)

print(preds)
calculate_score(preds)
