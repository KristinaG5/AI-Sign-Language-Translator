import cv2
import os
import numpy as np
from datetime import datetime
import subprocess

from preprocessing.run_preprocessing import extract_video
from application.predict import predict_sentence
from application.wireframe_video import generate_wireframe_video


def get_fps(video_path):
    """Uses OpenCV to get FPS of video

    Arguments
    ---------
    video_path : string
    Path to video

    Returns
    ---------
    FPS
    """
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Failed in opening video: {video_path}"
    return cap.get(cv2.CAP_PROP_FPS)


def get_timestamp():
    """Gets current timestamp

    Returns
    ---------
    Timestamp
    """
    return datetime.now().strftime("%d%m%Y_%H%M%S%f")


def seconds_to_timestamp(duration):
    """Converts duration into a timestamp format

    Arguments
    ---------
    duration : float
    Duration in seconds

    Returns
    ---------
    Timestamp
    """
    return datetime.fromtimestamp(duration).strftime("%M:%S.%f")[:-3]


def build_subtitles(subtitle_path, fps, predictions):
    """Produce a subtitle file

    Arguments
    ---------
    subtitle_path : string
    Path to save the subtitle

    fps : int
    FPS of the video

    predictions : array
    Array of predictions

    Returns
    ---------
    Path to the subtitles
    """
    with open(subtitle_path, "w") as f:
        f.write("WEBVTT\n\n")

        for prediction in predictions:
            start = prediction["start"] / fps
            end = prediction["end"] / fps
            word = prediction["word"]
            f.write(f"{seconds_to_timestamp(start)} --> {seconds_to_timestamp(end)}\n-{word}\n")

    return subtitle_path


def get_transcription_results(
    video_path,
    subtitle_path,
    hand_bbox_detector,
    body_mocap,
    hand_mocap,
    model,
    wireframe_video_path,
    min_score,
    min_confidence,
    scores_path,
    logging,
    ):

    """Produces the translation of each video

    Arguments
    ---------
    video_path : string
    Path to video

    subtitle_path : string
    Path to subtitle

    hand_bbox_detector, body_mocap, hand_mocap : Frankmocap Objects

    model : Model file (Pickle)

    wireframe_video_path : string
    Path to wireframe video

    min_score : int
    Minimum score threshold

    min_confidence : float
    Minimum confidence threshold

    scores_path : string
    Path to scores

    logging : Python logging object

    Returns
    ---------
    Produces wireframe video, scores and subtitles
    """
    extracted = extract_video(
        video_path,
        "frankmocap",
        "ratio",
        hand_bbox_detector=hand_bbox_detector,
        body_mocap=body_mocap,
        hand_mocap=hand_mocap,
    )

    logging.info("Predicting video labels...")
    normalised = np.array(extracted["normalised"])
    confidences = model.predict(normalised, logger=logging)
    predictions = predict_sentence(confidences, model.classes, min_confidence=min_confidence, min_score=min_score)

    if wireframe_video_path:
        generate_wireframe_video(video_path, extracted, wireframe_video_path)
    else:
        extension = video_path.split(".")[1]
        if extension != "mp4":
            converted_video_path = video_path.replace(extension, "mp4")
            subprocess.check_output(["ffmpeg", "-i", video_path, converted_video_path])

    fps = get_fps(video_path)
    subtitle_path = build_subtitles(subtitle_path, fps, predictions)

    with open(scores_path, "w") as f:
        f.write("word,score,start,end\n")
        for prediction in predictions:
            f.write(f"{prediction['word']},{prediction['score']},{prediction['start']/fps},{prediction['end']/fps}\n")

    logging.info("Complete")
