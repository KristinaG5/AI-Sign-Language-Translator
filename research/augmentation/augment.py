import json
import copy
import argparse


from research.augmentation.add_noise import add_noise, generate_noise


def augment(videos):
    augmented_videos = copy.deepcopy(videos)
    for video in videos:
        for augmentation in range(3):
            augmented_videos.append(generate_noise(video))
    return augmented_videos


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description="Adds noise to the coordinates to produce more data")
    parser.add_argument("-i", "--json_in", help="Name of the input json", type=str, required=True)
    parser.add_argument("-o", "--json_out", help="Name of the output json", type=str, required=True)
    args = parser.parse_args()

    # opens coordinates file
    with open(args.json_in) as json_file:
        videos = json.load(json_file)

    augmented_videos = augment(videos)

    with open(args.json_out, "w") as outfile:
        json_data = json.dumps(augmented_videos, indent=True)
        outfile.write(json_data)
        outfile.write("\n")
