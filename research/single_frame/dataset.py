import os
import json
import numpy as np


class Dataset:
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder
        self.data = self.load_jsons()
        self.sample_labels = []
        self.class_names = []
        self.model = None

    def load_json(self, path):
        with open(path) as json_file:
            videos = json.load(json_file)
        return videos

    def load_jsons(self):
        jsons = os.listdir(self.dataset_folder)
        data = {}
        for file_name in jsons:
            data[file_name[:-5]] = self.load_json(os.path.join(self.dataset_folder, file_name))
        return data

    def get_train_test_data(self, train_size):
        x_train = []
        y_train = []
        x_test = []
        y_test = []

        for class_name, videos in self.data.items():
            self.class_names.append(class_name)
            for video in videos:
                num_train_samples = round(len(video["normalised"]) * train_size)
                # TODO fix video:
                self.sample_labels.extend([(video["video:"], i) for i in range(0, num_train_samples)])
                x_train.extend(video["normalised"][:num_train_samples])
                x_test.extend(video["normalised"][num_train_samples:])
                y_train.extend([class_name] * num_train_samples)
                y_test.extend([class_name] * (len(video["normalised"]) - num_train_samples))

        x_train = [np.array(l) for l in x_train]
        x_test = [np.array(l) for l in x_test]
        return x_train, y_train, x_test, y_test

    def train(self, model, train_size):
        x_train, y_train, x_test, y_test = self.get_train_test_data(train_size)
        self.model = model
        self.model.fit(x_train, y_train)
        return self.model.score(x_test, y_test)

    def get_sample_label(self, index):
        if not self.model:
            raise Exception("Dataset not yet trained")
        return self.sample_labels[index]

    def predict(self, video):
        distances, indices = self.model.kneighbors(video)
        predictions = self.model.predict_proba(video)
        indices = indices.reshape(1, -1)[0]
        closest_samples = [self.get_sample_label(i) for i in indices]
        return predictions, closest_samples
