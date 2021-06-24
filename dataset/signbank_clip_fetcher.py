import requests
import os
from tqdm import tqdm

if __name__ == "__main__":
    """Downloads BSL Signbank videos from the url file produced

    Returns
    ---------
    A folder containing all of the downloaded videos
    """

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-o", "--output_folder", help="Name of the output folder to save the videos to", type=str, required=True
    )

    args = parser.parse_args()

    with open("word_video_url.txt", "r") as f:
        urls = f.readlines()

    for url in tqdm(urls):
        url = url.strip()
        r = requests.get(url)
        file_name = url.split("/")[-1]
        file_path = os.path.join(args.output_folder, file_name)
        open(file_path, "wb").write(r.content)
