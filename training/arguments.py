import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument("--folder_path", help="Path to folder of jsons", type=str, required=True)
    parser.add_argument(
        "--dtw_method",
        help="Dtw calculation method, default is standard",
        type=str,
        default="standard",
        choices=["standard", "fastdtw"],
    )
    parser.add_argument("--n_neighbors", help="N neighbors, default is 5", type=int, default=5)
    parser.add_argument("--max_warping_window", help="Max warping window, default is 10", type=int, default=10)
    parser.add_argument("--evaluate", help="Should evaluate and score model, default is True", type=bool, default=True)
    parser.add_argument("--train_size", help="Train test split, default is 0.8", type=float, default=0.8)

    # output parameters
    parser.add_argument("--model_path", help="Path to output model", type=str, required=True)

    args = parser.parse_args()

    return args
