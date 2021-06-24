import os
import json


def get_video_ranges(path):
    counter = 0
    ranges = {}

    for filename in os.listdir(path):
        with open(os.path.join(path, filename)) as f:
            videos = json.load(f)
            for video in videos:
                length = len(video["body"])
                ranges[video["video:"]] = (counter, counter + length)
                counter += length

    return ranges


def get_comparison_frame(indices):
    path = "data/frankmocap/unnormalised_ratio/"
    ranges = get_video_ranges(path)
    results = []

    for index in indices:
        for video_name, r in ranges.items():
            if index >= r[0] and index < r[1]:
                frame_number = index - r[0]
                results.append((video_name, frame_number))

    return results


print(get_comparison_frame([5, 50, 100, 500]))
