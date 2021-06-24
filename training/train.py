import os
import json
import numpy as np
import pickle
from sklearn.metrics import classification_report
import sys
from os.path import dirname, abspath

sys.path.append(dirname(dirname(abspath(__file__))))


from training.arguments import get_arguments
from training.dynamic_time_warping import KnnDtw


def load_json(path):
    """Loads single video from JSON

    Arguments
    ---------
    path : string
        Path to JSON folder

    Returns
    -------
    JSON content
    """
    with open(path) as json_file:
        videos = json.load(json_file)
    return videos


def load_jsons(dataset_folder):
    """Load JSONs

    Arguments
    ---------
    dataset_folder : string
        Path to JSON folder

    Returns
    -------
    Dictionary containing class names used in JSON filename
    """
    jsons = os.listdir(dataset_folder)
    data = {}
    for file_name in jsons:
        data[file_name[:-5]] = load_json(os.path.join(dataset_folder, file_name))
    return data


def get_labels(dataset_folder):
    """Retrieves training labels

    Arguments
    ---------
    dataset_folder : string
        Path to JSON folder

    Returns
    -------
    Array of labelled classes

    """
    class_names = [file_name[:-5] for file_name in os.listdir(dataset_folder)]
    return class_names


def train(model, folder_path, model_output_path, evaluate, train_size=None):
    """Main training function

    Arguments
    ---------
    model : KnnDtw
        Using KNN and DTW for training

    folder_path : string
        Path to preprocessed JSONs

    model_output_path : string
        Desired path to save weights

    evaluate : boolean
        Show classification report

    train_size : float
        Size of train test split
    """
    data = load_jsons(folder_path)

    if evaluate:
        train_x = []
        train_y = []
        test_x = []
        test_y = []

        for class_name, videos in data.items():
            for video in videos:
                num_train_samples = int(train_size * len(video["normalised"]))
                for frame in video["normalised"][:num_train_samples]:
                    train_x.append(frame)
                    train_y.append(model.class_to_num[class_name])

                for frame in video["normalised"][num_train_samples:]:
                    test_x.append(frame)
                    test_y.append(model.class_to_num[class_name])

        train_x = np.array(train_x)
        train_y = np.array(train_y)
        test_x = np.array(test_x)
        test_y = np.array(test_y)

        model.fit(train_x, train_y)
        confidences = model.predict(test_x)
        predictions = np.argmax(confidences, axis=1)
        print(classification_report(predictions, test_y, target_names=[l for l in model.classes]))

    else:
        x = []
        y = []
        for class_name, videos in data.items():
            for video in videos:
                for frame in video["normalised"]:
                    x.append(frame)
                    y.append(model.class_to_num[class_name])

        x = np.array(x)
        y = np.array(y)

        model.fit(x, y)

    pickle.dump(model, open(model_output_path, "wb"))


if __name__ == "__main__":
    """
    Training script using KNN and DTW
    """
    args = get_arguments()

    labels = get_labels(args.folder_path)
    print(labels)
    model = KnnDtw(args.dtw_method, labels, args.n_neighbors, args.max_warping_window)
    train(model, args.folder_path, args.model_path, args.evaluate, args.train_size)
