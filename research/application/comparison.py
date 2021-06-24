import cv2
import json
import pickle
import argparse
import numpy as np
import os

from pose_estimation.extract_pose import extract_video
from pose_estimation.openpose_pose_estimation import load_openpose, estimate_pose
from pose_estimation.frankmocap_pose_estimation import load_frankmocap, run_frank_mocap
from normalising.normalise import normalise_video


FRAME_RATE = 25
class_names = ["car", "cat", "dog", "i", "mechanic", "my", "story", "true"]
# class_names = ["i", "true"]

op = None
opWrapper = None
hand_bbox_detector = None
body_mocap = None
hand_mocap = None
sample_rate = None


# def get_video_ranges(path):
#     counter = 0
#     ranges = {}

#     # for filename in os.listdir(path):
#     #     with open(os.path.join(path, filename)) as f:
#     #         videos = json.load(f)
#     for video in dataset.data
#             # for video in videos:
#                 length = round(len(video['body'])*0.8)
#                 # print(video['video:'], length)
#                 ranges[video['video:']] = (counter, counter+length-1)
#                 counter += length

#     return ranges

# def get_comparison_frame(indices):
#     path = "data/frankmocap/ratio/"
#     ranges = get_video_ranges(path)
#     print(ranges)
#     results = []

#     for index in indices:
#         for video_name, r in ranges.items():
#             if index >= r[0] and index <= r[1]:
#                 frame_number = index - r[0]
#                 results.append((video_name, frame_number))

#     return results


def predict(video_path, model, method, normalisation_type):
    if method == "openpose":
        pose = extract_video(video_path, op, opWrapper, sample_rate, show_video=False)
    else:
        pose = run_frank_mocap(video_path, hand_bbox_detector, body_mocap, hand_mocap)

    normalised = normalise_video(pose, normalisation_type)
    normalised = [np.array(a) for a in normalised]
    distances, indices = model.kneighbors(normalised)
    predictions = model.predict_proba(normalised)
    indices = indices.reshape(1, -1)[0]
    comparison = get_comparison_frame(indices)
    return predictions, comparison


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


def get_frame_of_video(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    cap.set(1, frame_number)
    res, frame = cap.read()
    return frame


def app(video, model, pose_estimation_library, normalisation_type):
    global class_names, op, opWrapper, hand_bbox_detector, body_mocap, hand_mocap, sample_rate

    with open("data/validation/translate_bf3n.txt") as f:
        labels = f.readlines()

    clf = pickle.load(open(model, "rb"))
    predictions, comparison = predict(video, clf, pose_estimation_library, normalisation_type)

    validation_video_cap = cv2.VideoCapture(video)

    scores = []
    counter = 0
    while validation_video_cap.isOpened():
        ret, frame = validation_video_cap.read()
        if frame is None:
            break

        if counter % sample_rate == 0:
            index = int(counter / sample_rate)
            pred = predictions[index]
            comparison_video, comparison_frame = comparison[index]
            print(comparison_video, comparison_frame)
            actual = get_label(counter, labels, frame)
            confidence = get_prediction_confidence(pred, actual)
            scores.append(confidence)
            label_predictions(pred, class_names, frame)
            cv2.imshow("frame", frame)
            cv2.imshow("comparison", get_frame_of_video(comparison_video, comparison_frame))

            if cv2.waitKey(0) & 0xFF == ord("q"):
                break

        counter += 1

    # print("SCORE =", sum(scores) / len(scores))

    validation_video_cap.release()
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
    parser.add_argument("-m", "--openpose_models_dir", help="Path to the openpose models dir", type=str, required=False)
    args = parser.parse_args()

    if args.pose_estimation_library == "openpose":
        sample_rate = args.sample_rate
        op, opWrapper = load_openpose(args.openpose_dir, args.openpose_models_dir)
    else:
        sample_rate = 1
        hand_bbox_detector, body_mocap, hand_mocap = load_frankmocap(args.frankmocap_dir)

    app(args.video, args.weights, args.pose_estimation_library, args.normalisation_type)
