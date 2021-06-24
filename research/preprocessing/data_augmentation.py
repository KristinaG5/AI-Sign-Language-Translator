import sys
import cv2
import os
from sys import platform
import argparse
import copy
import json

# open absolute_coords.json to get the absolute coordinates
with open("absolute_coords.json") as json_file:
    data = json.load(json_file)

dict_copy = copy.deepcopy(data)


def mirror(items):
    for item in items:
        x = item[0]
        # formula to calculate how many pixels to move point by -(x-mp)*2
        formula = -(x - half_width) * 2
        item[0] = x + formula
    return items


# augment data to be inverted
for key in dict_copy.keys():
    augemented_video = dict_copy[key]
    half_width = augemented_video["width"] / 2
    body = augemented_video["body"]

    body = mirror(body)

    data[key + "_inverted"] = augemented_video

# augment data to be scaled
for key in dict_copy.keys():
    augemented_video = dict_copy[key]

    data[key + "_scaled"] = augemented_video

with open("augmented.json", "w") as json_file:
    data = json.dumps(data, indent=True)
    json_file.write(data)
    json_file.write("\n")
