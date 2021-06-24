import pickle
import numpy as np
import cv2
from pose_estimation.frankmocap_pose_estimation import load_frankmocap, run_frank_mocap
from normalising.normalise import normalise_video

hand_bbox_detector, body_mocap, hand_mocap = load_frankmocap("pose_estimation/frankmocap/extra_data/")


def normalise(video_path, normalisation_type):
    pose = run_frank_mocap(video_path, hand_bbox_detector, body_mocap, hand_mocap)
    normalised = normalise_video(pose, normalisation_type)
    normalised = [np.array(a) for a in normalised]
    return normalised


def get_frame_of_video(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    cap.set(1, frame_number)
    res, frame = cap.read()
    return frame


dataset = pickle.load(open("dataset_obj.pickle", "rb"))
print(dataset.model)

video = "data/validation/BF3n_9575.mov"
normalised = normalise(video, "ratio")
predictions, closest_samples = dataset.predict(normalised)
print(predictions)
validation_video_cap = cv2.VideoCapture(video)

sample_rate = 5
counter = 0
while validation_video_cap.isOpened():
    ret, frame = validation_video_cap.read()
    if frame is None:
        break

    if counter % sample_rate == 0:
        print(counter)
        pred = predictions[counter]
        comparison_video, comparison_frame = closest_samples[counter]
        print(comparison_video, comparison_frame)
        cv2.imshow("frame", frame)
        cv2.imshow("comparison", get_frame_of_video(comparison_video, comparison_frame))

        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

    counter += 1

validation_video_cap.release()
cv2.destroyAllWindows()
