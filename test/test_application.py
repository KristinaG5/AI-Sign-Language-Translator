import sys
from os.path import dirname, abspath

sys.path.append(dirname(dirname(abspath(__file__))))
import os
from application.predict import predict_sentence
from application.utils import build_subtitles, get_fps

# Application test script to test the post-processing steps of generating the subtitle file and predicted sentence


def test_predict_sentence():
    confidences = [
        [0.2, 0.6, 0.1, 0.1],
        [0.3, 0.5, 0.1, 0.1],
        [0.2, 0.5, 0.0, 0.3],
        [0.1, 0.1, 0.5, 0.3],
        [0.1, 0.1, 0.3, 0.5],
        [0.0, 0.1, 0.8, 0.1],
        [0.8, 0.1, 0.1, 0.0],
        [0.1, 0.2, 0.7, 0.0],
        [0.0, 0.1, 0.5, 0.4],
        [0.1, 0.1, 0.0, 0.8],
    ]
    classes = ["dog", "cat", "bird", "rabbit"]
    predictions = predict_sentence(confidences, classes, min_confidence=0.2, min_score=1, min_separation=2)
    assert predictions == [
        {"word": "cat", "score": 1.6, "start": 0, "end": 2},
        {"word": "rabbit", "score": 1.1, "start": 2, "end": 3},
        {"word": "bird", "score": 2.8, "start": 3, "end": 8},
        {"word": "rabbit", "score": 1.2, "start": 8, "end": 9},
    ]


def test_get_fps():
    video_path = os.path.join("files", "validation", "cat_1.mp4")
    fps = get_fps(video_path)
    assert fps == 30


def test_build_subtitles():
    fps = 25
    predictions = [
        {"word": "cat", "score": 1.8, "start": 0, "end": 15},
        {"word": "bird", "score": 1.7, "start": 25, "end": 40},
    ]
    subtitle_path = "subtitle.vtt"

    build_subtitles(subtitle_path, fps, predictions)

    with open(subtitle_path) as f:
        text = f.read()

    assert text == "WEBVTT\n\n00:00.000 --> 00:00.600\n-cat\n00:01.000 --> 00:01.600\n-bird\n"

    os.remove(subtitle_path)
