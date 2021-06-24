import cv2
import json
import pickle
import argparse
import numpy as np

from pose_estimation.extract_pose import extract_video
from pose_estimation.openpose_pose_estimation import load_openpose, estimate_pose
from pose_estimation.frankmocap_pose_estimation import load_frankmocap, run_frank_mocap
from normalising.normalise import normalise_video


FRAME_RATE = 25
class_names = ["story", "true", "car", "i", "my", "mechanic"]
# class_names = ["i", "true"]

op = None
opWrapper = None
hand_bbox_detector = None
body_mocap = None
hand_mocap = None
sample_rate = None


def predict(video_path, model, method, normalisation_type):
    # with open("data/validation/video.json") as f:
    #     pose = json.load(f)
    # pose = extract_video(video_path, op, opWrapper, sample_rate, show_video=False)
    if method == "openpose":
        pose = extract_video(video_path, op, opWrapper, sample_rate, show_video=False)
    else:
        pose = run_frank_mocap(video_path, hand_bbox_detector, body_mocap, hand_mocap)
    normalised = normalise_video(pose, normalisation_type)
    normalised = [np.array(a) for a in normalised]
    distances, indices = model.kneighbors(normalised)
    print(distances, indices)
    return model.predict_proba(normalised)


def label_predictions(predictions, class_names, frame):
    for i in range(len(predictions)):
        cv2.putText(
            frame,
            f"{class_names[i]}: {int(predictions[i]*100)}%",
            (10, 50 + (35 * i)),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )


def get_prediction_confidence(predictions, word):
    for i in range(len(predictions)):
        if class_names[i].lower() == word.lower():
            return predictions[i]


def get_label(frame_no, labels, frame):
    current_second = frame_no / FRAME_RATE
    for label in labels:
        word, start, end = label.strip().split(",")
        if current_second >= float(start) and current_second <= float(end):
            cv2.putText(
                frame,
                word,
                (10, 50 + (35 * len(class_names) + 1)),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            return word


def app(video, model, pose_estimation_library, normalisation_type):
    global class_names, op, opWrapper, hand_bbox_detector, body_mocap, hand_mocap, sample_rate
    cap = cv2.VideoCapture(video)
    clf = pickle.load(open(model, "rb"))
    predictions = predict(video, clf, pose_estimation_library, normalisation_type)
    print(predictions)
    scores = []
    with open("data/validation/translate_bf3n.txt") as f:
        labels = f.readlines()

    counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break

        if counter % sample_rate == 0:
            pred = predictions[int(counter / sample_rate)]
            actual = get_label(counter, labels, frame)
            confidence = get_prediction_confidence(pred, actual)
            scores.append(confidence)
            label_predictions(pred, class_names, frame)
            cv2.imshow("frame", frame)

            if cv2.waitKey(0) & 0xFF == ord("q"):
                break

        counter += 1

    print("SCORE =", sum(scores) / len(scores))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-v", "--video", help="Video ", type=str, required=True)
    parser.add_argument("-w", "--weights", help="Model", type=str, required=True)
    parser.add_argument("-s", "--sample_rate", help="Choose sample rate, default is 15", type=int, default=15)
    parser.add_argument(
        "-n", "--normalisation_type", help="Pick which normalisation technique to use", type=str, default="ratio"
    )
    parser.add_argument(
        "-t",
        "--pose_estimation_library",
        help="Choose which pose estimation library to use to extract the pose",
        type=str,
        default="openpose",
    )
    parser.add_argument("-x", "--frankmocap_dir", help="Path to the frankmocap models", type=str, required=False)
    parser.add_argument("-p", "--openpose_dir", help="Path to the openpose build dir", type=str, required=False)
    args = parser.parse_args()

    if args.pose_estimation_library == "openpose":
        sample_rate = args.sample_rate
        op, opWrapper = load_openpose(args.openpose_dir, args.openpose_models_dir)
    else:
        sample_rate = 1
        hand_bbox_detector, body_mocap, hand_mocap = load_frankmocap(args.frankmocap_dir)

    app(args.video, args.weights, args.pose_estimation_library, args.normalisation_type)
