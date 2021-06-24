import json
import pickle
import numpy as np
import os


with open("../../data/validation_ratio.json") as f:
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

X = []
num_samples = []

for video in videos:
    counter = 0
    for frame in video["normalised"]:
        X.append(frame)
        counter += 1
    num_samples.append(counter)


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


X = [np.array(l) for l in X]
model_folder = "weights"
for classifier in os.listdir(model_folder):
    clf = pickle.load(open(os.path.join(model_folder, classifier), "rb"))
    pred = clf.predict(X)
    print(classifier)
    calculate_score(pred)
    print()
