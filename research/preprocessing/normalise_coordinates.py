import json
import argparse
import math


def normalise(pointA, pointB):
    x1, y1 = pointA
    x2, y2 = pointB

    dx = x1 - x2
    dy = y1 - y2
    length = math.sqrt((dx * dx) + (dy * dy))

    return dx / length, dy / length


if __name__ == "__main__":
    """
    Normalises the coordinate data by finding the change in x,y and saving it to a new json file
    """

    # Arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--input_json", help="", type=str, required=True)
    parser.add_argument("-o", "--output_json", help="", type=str, required=True)
    args = parser.parse_args()

    # opens coordinates file
    with open(args.input_json) as json_file:
        data = json.load(json_file)

    output = {}

    for key, value in data.items():
        body = value["body"]
        normalised_frames = []

        for frame in body:
            nose = frame[0]
            neck = frame[1]
            rshoulder = frame[2]
            relbow = frame[3]
            rwrist = frame[4]
            lshoulder = frame[5]
            lelbow = frame[6]
            lwrist = frame[7]

            frame_points = [
                normalise(nose, neck),
                normalise(neck, rshoulder),
                normalise(neck, lshoulder),
                normalise(rshoulder, relbow),
                normalise(lshoulder, lelbow),
                normalise(relbow, rwrist),
                normalise(lelbow, lwrist),
            ]
            normalised_frames.append(frame_points)

        output[key] = normalised_frames

    with open(f"{args.output_json}.json", "w") as outfile:
        json_data = json.dumps(output, indent=True)
        outfile.write(json_data)
        outfile.write("\n")
