import cairosvg
import cv2
from PIL import Image
from io import BytesIO
import numpy as np
import json
import argparse

# takes json file and converts it to svg coords
def body_to_svg(nose, neck, rshoulder, relbow, rwrist, lshoulder, lelbow, lwrist):
    line = f'<line x1="{nose[0]}" y1="{nose[1]}" x2="{neck[0]}" y2="{neck[1]}" style="stroke:rgb(255,0,0);stroke-width:2" />'
    line += f'<line x1="{neck[0]}" y1="{neck[1]}" x2="{rshoulder[0]}" y2="{rshoulder[1]}" style="stroke:rgb(255,0,0);stroke-width:2" />'
    line += f'<line x1="{rshoulder[0]}" y1="{rshoulder[1]}" x2="{relbow[0]}" y2="{relbow[1]}" style="stroke:rgb(255,0,0);stroke-width:2" />'
    line += f'<line x1="{relbow[0]}" y1="{relbow[1]}" x2="{rwrist[0]}" y2="{rwrist[1]}" style="stroke:rgb(255,0,0);stroke-width:2" />'
    line += f'<line x1="{neck[0]}" y1="{neck[1]}" x2="{lshoulder[0]}" y2="{lshoulder[1]}" style="stroke:rgb(255,0,0);stroke-width:2" />'
    line += f'<line x1="{lshoulder[0]}" y1="{lshoulder[1]}" x2="{lelbow[0]}" y2="{lelbow[1]}" style="stroke:rgb(255,0,0);stroke-width:2" />'
    line += f'<line x1="{lelbow[0]}" y1="{lelbow[1]}" x2="{lwrist[0]}" y2="{lwrist[1]}" style="stroke:rgb(255,0,0);stroke-width:2" />'
    return line


# function to draw a rectangle around the skeleton
def draw_rectangle(x_min, x_max, y_min, y_max):
    # x_min draws left line
    # y_min draws top line
    # x_max draws right line
    # y_max draws bottom line

    width = x_max - x_min
    height = y_max - y_min
    line = f'<rect x="{x_min}" y="{y_min}" width="{width}" height="{height}" style="stroke:green;stroke-width:2;fill-opacity:0.1;stroke-opacity:0.9" />'
    return line


# gets the pixel coordinate per frame
def get_coords(value, frame):
    # value: 0 for x, 1 for y coord
    coord_values = [i[value] for i in frame]
    coord_sorted = sorted(coord_values)
    coord_min = coord_sorted[0]
    coord_max = coord_sorted[-1]
    return coord_min, coord_max


# shows the svg
def display_svg(svg):
    png = cairosvg.svg2png(bytestring=svg)
    img = Image.open(BytesIO(png))
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
    cv2.imshow("frame", cv_img)
    while True:
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    """
    Visualises the coordinates extracted as SVG
    """

    # Arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-j", "--json", help="Name of the input json that has been normalised and augmented", type=str, required=True
    )
    parser.add_argument(
        "-r",
        "--rectangle",
        help="Boolean to display a rectangle around the wireframe",
        type=bool,
        required=False,
        default=False,
    )
    args = parser.parse_args()

    # opens coordinates file
    with open(args.json) as json_file:
        data = json.load(json_file)

    width = 1000
    height = 1000
    counter = 0
    for frame in data:
        frame = [(item[0] * 1000, item[1] * 1000) for item in frame]
        nose = frame[0]
        neck = frame[1]
        rshoulder = frame[2]
        relbow = frame[3]
        rwrist = frame[4]
        lshoulder = frame[5]
        lelbow = frame[6]
        lwrist = frame[7]

        counter += 1

        # x_min, x_max = get_coords(0, frame)
        # y_min, y_max = get_coords(1, frame)

        body = body_to_svg(nose, neck, rshoulder, relbow, rwrist, lshoulder, lelbow, lwrist)
        # rect = draw_rectangle(x_min, x_max, y_min, y_max)
        svg = f'<svg height="{height}" width="{width}"> {body}'
        if args.rectangle:
            svg += rect
        svg += "</svg>"
        display_svg(svg)
