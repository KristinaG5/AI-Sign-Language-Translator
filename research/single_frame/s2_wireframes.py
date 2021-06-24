import sys
import cv2
import os
from sys import platform
import argparse
import json

if __name__ == "__main__":
    """
    Converts frames into coordinates
    """

    # Flags
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-f", "--folder_path", help="Path to folder where the video frames are located", type=str, required=True
    )
    parser.add_argument("-o", "--output_json", help="Name of the output json", type=str, required=True)
    parser.add_argument("-p", "--openpose_dir", help="Path to the openpose build dir", type=str, required=True)
    parser.add_argument("-m", "--openpose_models_dir", help="Path to the openpose models dir", type=str, required=True)
    parser.add_argument("-v", "--show_video", help="Boolean to show video", type=bool, default=False)
    parser.add_argument("-s", "--sample_rate", help="Choose sample rate, default is 15", type=int, default=15)
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

    # Process frames
    video_files = os.listdir(args.folder_path)
    data = []

    if video_files:
        for video in video_files:
            print(video)
            item = {"width": 0, "height": 0, "body": []}
            video_path = os.path.join(args.folder_path, video)
            cap = cv2.VideoCapture(video_path)
            width = cap.get(3)
            height = cap.get(4)

            counter = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if frame is None:
                    break
                item["width"] = width
                item["height"] = height
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
                    valid = True
                    for i in range(0, 8):
                        points = (float(datum.poseKeypoints[0][i][0]), float(datum.poseKeypoints[0][i][1]))

                        # Checks if the coordinates are readable
                        if points[0] == 0 and points[1] == 0:
                            valid = False
                            break
                        frame_points.append(points)

                    if valid:
                        item["body"].append(frame_points)
                        data.append(item)

                        if args.show_video:
                            cv2.imshow("frame", datum.cvOutputData)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                break
                    else:
                        print("Skipping due to error")

                counter += 1

    cap.release()
    cv2.destroyAllWindows()

    with open(args.output_json, "w") as outfile:
        json_data = json.dumps(data, indent=True)
        outfile.write(json_data)
        outfile.write("\n")

    sys.exit(-1)
