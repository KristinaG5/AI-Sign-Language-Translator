import pickle
import json
import numpy as np
from application.predict_sentence import predict_sentence
from tqdm import tqdm

model = pickle.load(open("training/weights/standard_reduced_sentence.pickle", "rb"))
words = {
    1: "go",
    2: "have",
    3: "football",
    4: "look",
    5: "small",
    6: "awful",
    7: "alright",
    8: "but",
    9: "good",
    10: "home",
    11: "one",
    12: "thought",
    13: "i",
    14: "before",
    15: "number",
    16: "always",
    17: "finish",
    18: "when",
    19: "five",
    20: "children",
    21: "easter",
    22: "beat",
    23: "better",
    24: "summer",
    25: "laugh",
    26: "winter",
    27: "spring",
    28: "about",
    29: "remember",
}

with open("data/val_videos/validation_json/validation_videos.json") as f:
    data = json.loads(f.read())


model.max_warping_window = 15
video_results = {}
for video in tqdm(data):
    normalised = np.array(video["normalised"])
    labels, proba = model.predict(normalised)

    results = []
    last_label = labels[0]
    probabilities = [proba[0]]

    for i in range(1, len(labels)):
        label = labels[i]
        probability = proba[i]
        if label == last_label:
            probabilities.append(probability)
        else:
            if len(probabilities) > 1:
                results.append({"word": words[last_label], "count": len(probabilities), "proba": probabilities})
            last_label = label
            probabilities = [probability]

    video_results[video["video"]] = results

with open("mww15_results.txt", "w") as f:
    for name, data in video_results.items():
        f.write("\n" + name + "\n")
        for result in data:
            f.write(str(result) + "\n")
