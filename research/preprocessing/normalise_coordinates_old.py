import cairosvg
import cv2
from PIL import Image
from io import BytesIO
import numpy as np
import json
import argparse


def calculate_ratio(width, height, x_min, y_min, point_coord):
    x = point_coord[0] - x_min
    y = point_coord[1] - y_min
    x_percent = x / width
    y_percent = y / height
    return x_percent, y_percent


def get_coords(value, frame):
    # value: 0 for x, 1 for y coord
    coord_values = [i[value] for i in frame]
    coord_sorted = sorted(coord_values)
    coord_min = coord_sorted[0]
    coord_max = coord_sorted[-1]
    return coord_min, coord_max


if __name__ == "__main__":
    """
    Normalises the coordinate data by drawing a box around the wireframe and then scaling the points to ratios
    """

    # Arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--json", help="", type=str, required=True)
    parser.add_argument("-v", "--video", help="", type=str, required=True)
    args = parser.parse_args()

    # opens coordinates file
    with open(args.json) as json_file:
        data = json.load(json_file)

    video = data[args.video]
    body = video["body"]
    ratio_data = {}
    ratio_data[args.video] = {"body": []}

    for frame in body:
        x_min, x_max = get_coords(0, frame)
        y_min, y_max = get_coords(1, frame)

        box_width = x_max - x_min
        box_height = y_max - y_min

        frame_points = []
        for i in range(len(frame)):
            x, y = calculate_ratio(box_width, box_height, x_min, y_min, frame[i])
            frame_points.append((float(x), float(y)))
        ratio_data[args.video]["body"].append(frame_points)

    with open("ratio_coords.json", "w") as outfile:
        json_data = json.dumps(ratio_data, indent=True)
        outfile.write(json_data)
        outfile.write("\n")
