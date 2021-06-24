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


def load_data(path):
    with open(path) as json_file:
        frames = json.load(json_file)

    # flatten array
    out = []
    for frame in frames:
        x = np.array(frame)
        x = x.reshape((1, 16))[0]
        out.append(x)
    return out


if __name__ == "__main__":
    """
    Training
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

    # Load preprocessed data
    cat = load_data("train_cat.json")
    dog = load_data("train_dog.json")

    X = cat + dog
    Y = ["cat" for i in range(len(cat))] + ["dog" for i in range(len(dog))]
    assert len(X) == len(Y), "X and Y size do not match"

    # Train-test split
    train_X, test_X, train_Y, test_Y = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, random_state=1)

    # Train the model
    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        clf.fit(train_X, train_Y)
        score = clf.score(test_X, test_Y)
        print(name, score)

    predict_cat = load_data("train_predict_cat.json")
    x = clf.predict(predict_cat)
    print(x)
