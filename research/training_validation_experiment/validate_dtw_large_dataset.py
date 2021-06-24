import json
import pickle
import numpy as np
import os
import sys
from os.path import dirname, abspath

sys.path.append(dirname(dirname(dirname(abspath(__file__)))))
import training.train

with open("data/validation_large_dataset/validation_json/validation_videos.json") as f:
    videos = json.load(f)

video_labels = {
    "G19n_121100": [
        {"word": "go", "start": 44, "end": 55},
        {"word": "home", "start": 56, "end": 62},
        {"word": "spring", "start": 63, "end": 84},
        {"word": "easter", "start": 85, "end": 125},
        {"word": "summer", "start": 141, "end": 180},
        {"word": "winter", "start": 198, "end": 233},
    ],
    "BF23n_249811": [
        {"word": "i", "start": 0, "end": 9},
        {"word": "have", "start": 10, "end": 13},
        {"word": "five", "start": 14, "end": 19},
        {"word": "children", "start": 20, "end": 35},
    ],
    "G24n_31200": [
        {"word": "thought", "start": 0, "end": 1},
        {"word": "i", "start": 2, "end": 5},
        {"word": "better", "start": 11, "end": 17},
        {"word": "go", "start": 20, "end": 26},
        {"word": "home", "start": 26, "end": 36},
    ],
    "G14n_663035": [{"word": "one", "start": 0, "end": 19}],
    "G27n_2172": [
        {"word": "i", "start": 5, "end": 7},
        {"word": "remember", "start": 8, "end": 18},
        {"word": "football", "start": 33, "end": 43},
        {"word": "before", "start": 23, "end": 31},
        {"word": "i", "start": 48, "end": 59},
    ],
    "G15n_91493": [
        {"word": "small", "start": 9, "end": 17},
        {"word": "number", "start": 18, "end": 23},
        {"word": "but", "start": 24, "end": 25},
        {"word": "good", "start": 29, "end": 32},
        {"word": "laugh", "start": 33, "end": 64},
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


clf = pickle.load(open("standard_large.pickle", "rb"))
clf.max_warping_window = 1
clf.n_neighbors = 5

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
