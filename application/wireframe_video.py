import cv2
import os


def get_pair(data, index):
    """Gets x and y pairs from JSON

    Arguments
    ---------
    data : array 
    Array of normalised coordinates

    Returns
    ---------
    Pairs of x and y
    """
    return (round(data[index * 2]), round(data[(index * 2) + 1]))


def generate_wireframe_video(video_path, data, output_path):
    """Produce an SVG wireframe ontop of video

    Arguments
    ---------
    video_path : string
    Path to video

    data : array
    Array of normalised coordinates

    output_path : string
    Path to save the wireframe video

    Returns
    ---------
    Creates new video with a wireframe overlaid
    """
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Failed in opening video: {video_path}"

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    assert frame_rate <= 60, f"{video_path} has an invalid frame rate of {frame_rate}"

    # Get video frames
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        frames.append(frame)
    cap.release()

    # Save new video
    with open(os.path.join("../", "preprocessing", "visualise.csv")) as f:
        labels = [x.strip().split(",") for x in f.readlines()[1:]]

    size = (int(data["width"]), int(data["height"]))
    fourcc = cv2.VideoWriter_fourcc(*"VP09")
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, size)
    for i in range(len(frames)):
        frame = frames[i]
        # Add lines
        for type, origin, destination in labels:
            frame = cv2.line(
                frame,
                get_pair(data[type][i], int(origin)),
                get_pair(data[type][i], int(destination)),
                (0, 255, 0) if type == "body" else (0, 0, 255),
                2,
            )
        out.write(frame)
    out.release()
