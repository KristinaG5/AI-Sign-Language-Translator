import os
import json
from unittest import mock
import sys
import shutil
from os.path import dirname, abspath

sys.path.append(dirname(dirname(abspath(__file__))))
from preprocessing.run_preprocessing import preprocess, get_ratio, calculate_joint_angles

# Pre-processing test script to test preprocessing and normalisation


@mock.patch("preprocessing.run_preprocessing.run_frank_mocap")
def test_run_preprocessing(run_frank_mocap):
    with open(os.path.join("files", "cat.json")) as f:
        video = json.load(f)
        normalise = get_ratio(video)

    run_frank_mocap.return_value = video
    data_folder = os.path.join("files", "videos")
    data = {
        class_name: len(os.listdir(os.path.join(data_folder, class_name))) for class_name in os.listdir(data_folder)
    }
    assert len(data) > 0, "No test data"
    output_folder = "output"

    if os.path.isdir(output_folder):
        shutil.rmtree(output_folder)

    preprocess(data_folder, output_folder, "frankmocap", "ratio")
    run_frank_mocap.assert_called()

    assert os.path.isdir(output_folder), "Output directory not created"

    for class_name, num_videos in data.items():
        json_path = os.path.join(output_folder, f"{class_name}.json")
        assert os.path.isfile(json_path), "Output json is not valid"

        with open(json_path) as f:
            json_data = json.load(f)
            assert len(json_data) == num_videos, "Json does not match amount of videos"
            for video in json_data:
                assert list(video.keys()) == [
                    "video",
                    "width",
                    "height",
                    "body",
                    "left_hand",
                    "right_hand",
                    "normalised",
                ], "Json does not contain the right keys"
                assert video["normalised"] == normalise, "Normalised failed"

    shutil.rmtree(output_folder)


def test_normalise_ratio():
    data = {
        "body": [[10, 10, 100, 100, 55, 55]],
        "left_hand": [[25, 50, 50, 75, 30, 70]],
        "right_hand": [[20, 80, 60, 40, 70, 30]],
    }

    normalised = get_ratio(data)
    assert normalised == [[0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 0.2, 0.8, 0.0, 1.0, 0.8, 0.2, 1.0, 0.0]]
