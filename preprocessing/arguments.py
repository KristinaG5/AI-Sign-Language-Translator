import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    """
    folder_path is a folder of folders where each subfolder represents a class:

    ├── parent_folder
        ├── cat
            ├── cat1.mp4
            ├── cat2.mp4
            └── cat3.mp4
        └── dog
            ├── dog1.mp4
            ├── dog2.mp4
            └── dog3.mp4

    outputs a json file for each class to output_dir:

    ├── parent_folder
        ├── cat.json
        └── dog.json
    """

    # input parameters
    parser.add_argument("--folder_path", help="Path to folder of classes", type=str, required=True)
    parser.add_argument(
        "--normalisation_type", help="Normalisation technique", type=str, choices=["ratio", "angle"], required=True
    )
    parser.add_argument(
        "--method",
        help="Pose estimation library to extract the pose",
        type=str,
        choices=["frankmocap", "openpose"],
        required=True,
    )
    parser.add_argument(
        "--frankmocap_dir", help="Path to frankmocap models", type=str, default="preprocessing/frankmocap/extra_data/"
    )
    parser.add_argument("--openpose_dir", help="Path to openpose build dir", type=str)
    parser.add_argument("--openpose_models_dir", help="Path to openpose models dir", type=str)
    parser.add_argument("--fps", help="Choose sample rate, default is 25", type=int, default=25)

    # output parameters
    parser.add_argument("--output_dir", help="Path to output directory", type=str, required=True)

    args = parser.parse_args()

    assert args.folder_path, "Must provide video or folder path"

    if args.method == "openpose":
        assert args.openpose_dir, "Must provide openpose directory"
        assert args.openpose_models_dir, "Must provide openpose model directory"
    else:
        assert args.frankmocap_dir, "Must provide frankmocap directory"

    return args
