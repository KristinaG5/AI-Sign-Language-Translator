# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import json

if __name__ == "__main__":
    """
    -command python extract_wireframe.py -i ~/Desktop/dataset/test -o test_coords -p ~/Desktop/openpose/build/python -m ~/Desktop/openpose/models
    """

    # Flags
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--folder_path", help="", type=str, required=True)
    parser.add_argument("-o", "--output_json", help="", type=str, required=True)
    parser.add_argument("-p", "--openpose_dir", help="", type=str, required=True)
    parser.add_argument("-m", "--openpose_models_dir", help="", type=str, required=True)
    parser.add_argument("-v", "--show_video", help="", type=bool, default=False)
    parser.add_argument("-s", "--sample_rate", help="", type=int, default=15)
    args = parser.parse_args()

    try:
        sys.path.append(args.openpose_dir)
        from openpose import pyopenpose as op
    except ImportError as e:
        print("Error: OpenPose library could not be found.")
        raise e

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = args.openpose_models_dir
    params["face"] = True
    params["hand"] = True

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process frames from multiple videos
    video_files = os.listdir(args.folder_path)
    data = {}

    if video_files:
        for video in video_files:
            print(video)
            data[video] = {"width": 0, "height": 0, "body": []}
            video_path = os.path.join(args.folder_path, video)
            cap = cv2.VideoCapture(video_path)
            width = cap.get(3)
            height = cap.get(4)
            counter = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if frame is None:
                    break
                data[video]["width"] = width
                data[video]["height"] = height
                if counter % args.sample_rate == 0:
                    datum = op.Datum()
                    datum.cvInputData = frame
                    opWrapper.emplaceAndPop([datum])

                    # nose = 0
                    # neck = 1
                    # right shoulder = 2
                    # right elbow = 3
                    # right wrist = 4
                    # left shoulder = 5
                    # left elbow = 6
                    # left wrist = 7

                    frame_points = []
                    for i in range(0, 8):
                        frame_points.append((float(datum.poseKeypoints[0][i][0]), float(datum.poseKeypoints[0][i][1])))

                    data[video]["body"].append(frame_points)

                    if args.show_video:
                        cv2.imshow("frame", datum.cvOutputData)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                counter += 1

    cap.release()
    cv2.destroyAllWindows()

    with open(f"{args.output_json}.json", "w") as outfile:
        json_data = json.dumps(data, indent=True)
        outfile.write(json_data)
        outfile.write("\n")

    sys.exit(-1)
