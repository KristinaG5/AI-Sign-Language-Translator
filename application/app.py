from flask import Flask, render_template, request, jsonify, send_file
import os
import io
import numpy as np
import pickle
import cv2
import logging
from flask_socketio import SocketIO
import csv
import sys
from os.path import dirname, abspath

sys.path.append(dirname(dirname(abspath(__file__))))

from preprocessing.frankmocap_pose_estimation import load_frankmocap
from application.utils import get_timestamp, get_transcription_results

# App
app = Flask(__name__, template_folder="static")
socketio = SocketIO(app)
thread = None

# Models
hand_bbox_detector, body_mocap, hand_mocap = load_frankmocap(
    os.path.abspath(os.path.join("..", "preprocessing", "frankmocap", "extra_data"))
)
model = pickle.load(open(os.path.join("..", "standard_large.pickle"), "rb"))
model.max_warping_window = 1
model.n_neighbors = 5

uploads = os.path.join("static", "uploads")
subtitles = os.path.join("static", "subtitles")
scores = os.path.join("static", "scores")

os.makedirs(uploads, exist_ok=True)
os.makedirs(subtitles, exist_ok=True)
os.makedirs(scores, exist_ok=True)


# Logging
class ProgressLogger(logging.Handler):
    def emit(self, record):
        text = record.getMessage()
        print("LOG", text)
        if text.startswith("Progress"):
            text = text.split("-")[1]
            current, total = text.split("/")
            socketio.emit("progress", {"number": current, "total": total}, namespace="/app")
        elif text == "Complete":
            socketio.emit("complete", {}, namespace="/app")
        else:
            socketio.emit("message", {"text": text}, namespace="/app")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")
logger.addHandler(ProgressLogger())


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/", methods=["POST"])
def upload_file():
    global thread
    timestamp = get_timestamp()
    video_path = os.path.join("static", "uploads", timestamp + "." + request.files["file"].filename.split(".")[1])
    subtitle_path = os.path.join("static", "subtitles", timestamp + ".vtt")
    scores_path = os.path.join("static", "scores", timestamp + ".csv")

    if request.values.get("wireframe"):
        wireframe_video_path = os.path.join("static", "uploads", timestamp + "-wireframe.mp4")
    else:
        wireframe_video_path = None

    if request.values.get("minScore"):
        min_score = int(request.values.get("minScore"))

    if request.values.get("minConfidence"):
        min_confidence = float(request.values.get("minConfidence"))

    request.files["file"].save(video_path)

    thread = socketio.start_background_task(
        get_transcription_results,
        video_path=video_path,
        subtitle_path=subtitle_path,
        hand_bbox_detector=hand_bbox_detector,
        body_mocap=body_mocap,
        hand_mocap=hand_mocap,
        model=model,
        wireframe_video_path=wireframe_video_path,
        min_score=min_score,
        min_confidence=min_confidence,
        scores_path=scores_path,
        logging=logger,
    )
    return render_template("progress.html", id=timestamp)


@app.route("/results", methods=["GET"])
def results():
    id = request.args.get("id")
    video_path = os.path.join("static", "uploads", id + ".mp4")
    subtitle_path = os.path.join("static", "subtitles", id + ".vtt")
    wireframe_video_path = os.path.join("static", "uploads", id + "-wireframe.mp4")

    return render_template(
        "results.html",
        video_path=wireframe_video_path if os.path.isfile(wireframe_video_path) else video_path,
        subtitle_path=subtitle_path,
        id=id,
    )


@app.route("/scores", methods=["GET"])
def scores():
    id = request.args.get("id")
    scores_path = os.path.join("static", "scores", id + ".csv")
    with open(scores_path, newline="") as csvfile:
        data = list(csv.DictReader(csvfile))

    return render_template("scores.html", data=data, id=id)


if __name__ == "__main__":
    socketio.run(app)
