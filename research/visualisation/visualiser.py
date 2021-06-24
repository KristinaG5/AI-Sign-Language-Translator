import cairosvg
import cv2
from PIL import Image
from io import BytesIO
import numpy as np
import json
import argparse


def get_pair(data, index):
    return [data[index * 2], data[(index * 2) + 1]]


def body_to_svg(data, config_file):
    # takes json file and converts it to svg coords

    with open(config_file) as f:
        points = [line.strip() for line in f.readlines()]

    line = ""
    for point in points:
        origin_index, destination_index = point.split(",")
        origin = get_pair(data, int(origin_index))
        destination = get_pair(data, int(destination_index))
        line += f'<line x1="{origin[0]}" y1="{origin[1]}" x2="{destination[0]}" y2="{destination[1]}" style="stroke:rgb(255,0,0);stroke-width:2" />'

    return line


def display_svg(svg):
    png = cairosvg.svg2png(bytestring=svg)
    img = Image.open(BytesIO(png))
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
    cv2.imshow("frame", cv_img)
    while True:
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break


def visualise(size, data, config_file):
    for frames in data:
        for frame in frames:
            frame = [item * size for item in frame]
            body = body_to_svg(frame, config_file)
            display_svg(f'<svg height="{size}" width="{size}"> {body}</svg>')


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description="Visualises the coordinates extracted as SVG")
    parser.add_argument(
        "-j", "--json", help="Name of the input json that has been normalised and augmented", type=str, required=True
    )
    parser.add_argument("-c", "--config", help="Config file for joints", type=str, default="visualise.csv")
    parser.add_argument("-s", "--size", type=int, default=1000)
    args = parser.parse_args()

    # opens coordinates file
    with open(args.json) as json_file:
        data = json.load(json_file)

    visualise(args.size, data, args.config)
