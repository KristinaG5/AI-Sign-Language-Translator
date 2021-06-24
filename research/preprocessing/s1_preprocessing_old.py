import argparse
import json

from extract_wireframe import extract_wireframe
from normalise import normalise
from generate_noise import generate_noise

if __name__ == "__main__":
    """
    Runs preprocessing scripts to produce a final json to train on
    """
    parser = argparse.ArgumentParser(description="Apply preprocessing to videos folder")
    parser.add_argument(
        "-f", "--folder_path", help="Path to folder where the video frames are located", type=str, required=True
    )
    parser.add_argument("-o", "--output_json", help="Name of the output json", type=str, required=True)
    parser.add_argument("-p", "--openpose_dir", help="Path to the openpose build dir", type=str, required=True)
    parser.add_argument("-m", "--openpose_models_dir", help="Path to the openpose models dir", type=str, required=True)
    parser.add_argument("-v", "--show_video", help="Boolean to show video", type=bool, default=False)
    parser.add_argument("-s", "--sample_rate", help="Choose sample rate, default is 15", type=int, default=15)
    args = parser.parse_args()

    data = extract_wireframe(
        args.openpose_dir, args.openpose_models_dir, args.folder_path, args.sample_rate, args.show_video
    )
    normalise_data = normalise(data)
    noise_data = generate_noise(normalise_data)

    with open(args.output_json, "w") as outfile:
        json_data = json.dumps(noise_data, indent=True)
        outfile.write(json_data)
        outfile.write("\n")
