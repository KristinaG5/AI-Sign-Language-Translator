import cairosvg
import cv2
from PIL import Image
from io import BytesIO
import random
import json
import copy
import argparse


def generate_noise(points, num_augmentations, max_random):
    results = [points]
    random_range = max_random * 1000
    for i in range(num_augmentations):
        augmented = copy.deepcopy(points)
        for point in augmented:
            point[0] += random.randrange(-random_range, random_range + 1) / 1000
            point[1] += random.randrange(-random_range, random_range + 1) / 1000
        results.append(augmented)
    return results


if __name__ == "__main__":
    """
    Adds noise to the coordinates to produce more data
    """

    # Arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--json_in", help="Name of the input json", type=str, required=True)
    parser.add_argument("-o", "--json_out", help="Name of the output json", type=str, required=True)
    args = parser.parse_args()

    # opens coordinates file
    with open(args.json_in) as json_file:
        data = json.load(json_file)

    results = []
    for frame in data:
        results.extend(generate_noise(frame, 1, 0.05))

    with open(args.json_out, "w") as outfile:
        json_data = json.dumps(results, indent=True)
        outfile.write(json_data)
        outfile.write("\n")
