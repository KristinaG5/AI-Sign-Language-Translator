import cv2
import random
import json
import copy
import argparse
import numpy as np
import sklearn.model_selection
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from utils import get_training_data


def load_data(path):
    max_frames = 13
    num_values = max_frames * 16
    with open(path) as json_file:
        videos = json.load(json_file)
    out = []
    for video in videos:
        values = []
        for frame in video:
            for point in frame:
                values.append(point)
        if len(values) < num_values:
            values.extend([0] * (num_values - len(values)))

        out.append(np.array(values))
    return out


if __name__ == "__main__":
    """
    Training multi class
    """

    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
    ]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
    ]

    validation_size = 1
    data = get_training_data("../json/imbalance_classes")
    X = []
    Y = []
    validation_x = []
    validation_y = []

    # Load preprocessed data
    # TODO: pick validation samples at random
    for class_name, file_name in data.items():
        class_data = load_data(file_name)
        validation_x.extend(class_data[-validation_size:])
        validation_y.extend([class_name] * validation_size)
        X.extend(class_data[:-validation_size])
        Y.extend([class_name] * (len(class_data) - validation_size))

    print(validation_y)
    assert len(X) == len(Y), "X and Y size do not match"
    for class_name in data.keys():
        assert class_name in validation_y, f"No validation data for {class_name}"
        assert class_name in Y, f"No training data for {class_name}"
    assert len(validation_x) == len(validation_y), "validation X and Y size do not match"

    # Train-test split
    train_X, test_X, train_Y, test_Y = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, random_state=1)

    # Train the model
    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        clf.fit(train_X, train_Y)
        score = clf.score(test_X, test_Y)

        # Validate
        predictions = clf.predict(validation_x)
        print(predictions)
        validate_score = sum([predictions[i] == validation_y[i] for i in range(len(predictions))]) / len(validation_x)
        print("Name: " + str(name), "Accuracy: " + str(score), "Validation score: " + str(validate_score))
