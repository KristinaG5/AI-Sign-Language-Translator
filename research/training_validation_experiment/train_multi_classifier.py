import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import sklearn.model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pickle

from load_multi_frame import flatten_videos, pad_videos, pad_video
from load_single_frame import get_middle_frame

NAMES = [
    "Nearest Neighbors",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

CLASSIFIERS = [
    KNeighborsClassifier(3),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]


def load_json(path):
    with open(path) as json_file:
        videos = [v["normalised"] for v in json.load(json_file)]
    return videos


def load_data(json_folder, method):
    jsons = os.listdir(json_folder)
    X = []
    Y = []
    WINDOW_SIZE = 1500

    # Load preprocessed data
    if method == "multi":
        for file_name in jsons:
            class_data = load_json(os.path.join(json_folder, file_name))
            X.extend(class_data)
            Y.extend([file_name[:-5]] * len(class_data))

        X = flatten_videos(X)
        newX = []
        newY = []

        for i in range(len(X)):
            item = X[i]
            label = Y[i]
            for j in range(WINDOW_SIZE, len(item), 100):
                newX.append(item[j - WINDOW_SIZE : j])
                newY.append(label)
            newX.append(pad_video(item[j:], WINDOW_SIZE))
            newY.append(label)

        X = newX
        Y = newY
        print([len(x) for x in X], len(X))
    elif method == "single":
        for file_name in jsons:
            class_data = load_json(os.path.join(json_folder, file_name))
            for video in class_data:
                print(file_name, [len(f) for f in video])
                for frame in video:
                    X.append(frame)
                    Y.append(file_name[:-5])
    else:
        raise Exception("Invalid method")

    X = [np.array(l) for l in X]
    assert len(X) == len(Y), "X and Y size do not match"
    return X, Y


def train(json_folder, method, weights_path=None):
    X, Y = load_data(json_folder, method)
    class_names = [file_name[:-5] for file_name in os.listdir(json_folder)]

    # Train-test split
    train_X, test_X, train_Y, test_Y = sklearn.model_selection.train_test_split(
        X, Y, test_size=0.2, random_state=1, stratify=Y
    )

    for class_name in class_names:
        print(class_name, "has:", Y.count(class_name), "videos")
        assert class_name in test_Y, f"No test data for {class_name}"
        assert class_name in train_Y, f"No training data for {class_name}"

    # Train the model
    os.makedirs(weights_path)
    results = {}
    classifiers = list(zip(NAMES, CLASSIFIERS))

    for name, clf in tqdm(classifiers):
        clf.fit(train_X, train_Y)
        score = clf.score(test_X, test_Y)
        results[name] = score
        pickle.dump(clf, open(os.path.join(weights_path, name + ".pickle"), "wb"))

    return results


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-j", "--json_folder", help="Json folder that has been normalised and augmented", type=str, required=True
    )
    parser.add_argument("-w", "--weights_path", help="Path to save weights", type=str, default="")
    parser.add_argument("-m", "--method", help="Training technique", type=str, default="multi")
    args = parser.parse_args()

    results = train(args.json_folder, args.method, weights_path=args.weights_path)

    for name, score in results.items():
        print(f"{name}:{score}")
