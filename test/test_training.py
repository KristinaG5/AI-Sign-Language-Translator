import os
import pickle
import json
import numpy as np

from training.train import train, get_labels, KnnDtw

# Train test script to test training uses normalised data and DTW predict method


def test_train():
    dataset = os.path.join("files", "training_dataset")
    output_model = "model.pickle"

    labels = get_labels(dataset)
    assert labels == ["dog", "cat"]

    model = KnnDtw("standard", labels, 5, 1)
    train(model, dataset, output_model, False)

    assert os.path.isfile(output_model)
    imported_model = pickle.load(open(output_model, "rb"))
    assert imported_model

    with open(os.path.join("files", "training_dataset", "cat.json")) as f:
        video = json.load(f)[0]
        normalised = np.array(video["normalised"])

    confidences = model.predict(normalised)
    assert len(confidences) == len(normalised)
    for row in confidences:
        assert len(row) == len(labels)
        for item in row:
            assert item > 0 and item < 1

    os.remove(output_model)
