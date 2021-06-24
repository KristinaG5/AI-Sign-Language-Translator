import random
import copy


def add_noise(points, max_random):
    random_range = max_random * 1000
    for i in range(len(points)):
        points[i] += random.randrange(-random_range, random_range + 1) / 1000
        points[i] = min(points[i], 1.0)
        points[i] = max(points[i], 0.0)

    return points


def generate_noise(video):
    new_video = copy.deepcopy(video)
    new_frames = []
    for frame in new_video["normalised"]:
        new_frames.append(add_noise(frame, 0.05))

    new_video["normalised"] = new_frames
    return new_video
