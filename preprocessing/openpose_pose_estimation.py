import cv2
import sys


def load_openpose(openpose_dir, openpose_models_dir):
    """Loads OpenPose

    Arguments
    ---------
    openpose_dir : string
        Path to openpose build directory

    openpose_models_dir : string
        Path to openpose models directory

    Returns
    -------
    op, opWrapper
    """
    try:
        sys.path.append(openpose_dir)
        from openpose import pyopenpose as op
    except ImportError as e:
        print("Error: OpenPose library could not be found.")
        raise e

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = openpose_models_dir
    params["hand"] = 1

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    return op, opWrapper


def estimate_pose(op, opWrapper, frame, show_video):
    """Extract body, left and right hand pose estimation points

    Arguments
    ---------
    op, opWrapper : object
        OpenPose classifiers

    frame : image
        Single frame from video

    show_video : boolean
        Display image in window

    Returns
    -------
    Body, left and right hand key points
    """
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])
    body_frame_points = []
    left_hand_frame_points = []
    right_hand_frame_points = []

    for i in range(0, 8):
        body_frame_points.append(float(datum.poseKeypoints[0][i][0]))
        body_frame_points.append(float(datum.poseKeypoints[0][i][1]))

    for i in range(0, 21):
        right_hand_frame_points.append(float(datum.handKeypoints[0][0][i][0]))
        right_hand_frame_points.append(float(datum.handKeypoints[0][0][i][1]))
        left_hand_frame_points.append(float(datum.handKeypoints[1][0][i][0]))
        left_hand_frame_points.append(float(datum.handKeypoints[1][0][i][1]))

    if show_video:
        while True:
            cv2.imshow("frame", datum.cvOutputData)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    return body_frame_points, right_hand_frame_points, left_hand_frame_points


def run_openpose(op, opWrapper, video_path, fps):
    """Processes video through OpenCV to apply pose estimation

    Arguments
    ---------
    op, opWrapper : object
        OpenPose classifiers

    video_path : string
        Path to individual videos

    fps : int
        Rate at which to sample video

    Returns
    -------
    Dictionary
    """
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    assert frame_rate <= 60, f"{video_path} has an invalid frame rate of {frame_rate}"
    frame_sampling = round(frame_rate / fps)
    item = {
        "video": video_path,
        "width": cap.get(3),
        "height": cap.get(4),
        "body": [],
        "left_hand": [],
        "right_hand": [],
    }

    counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break

        if counter % frame_sampling == 0:
            try:
                body_frame_points, right_hand_frame_points, left_hand_frame_points = estimate_pose(
                    op, opWrapper, frame, show_video=False
                )
                item["body"].append(body_frame_points)
                item["right_hand"].append(right_hand_frame_points)
                item["left_hand"].append(left_hand_frame_points)
            except:
                cv2.imwrite(f"error_{video_path}_{counter}.png", frame)
                print(f"Failed to estimate {video_path}, frame {counter}")

        counter += 1

    cap.release()
    return item
