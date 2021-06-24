import requests
import os
import concurrent.futures
from itertools import repeat


def download(line, output_folder):
    """Downloads signbsl.com videos and renames them

    Arguments
    ---------
    line : string
    Individual video information from the txt file

    output_folder : string
    Path to save the videos

    Returns
    ---------
    A folder containing all of the downloaded videos
    """
    word, urls = line.split("|")
    urls = urls.strip()[1:-1].replace("'", "").split(", ")

    if len(urls) > 1:
        word_folder = os.path.join(output_folder, word)
        os.makedirs(word_folder, exist_ok=True)
        print(word)

        for i in range(len(urls)):
            url = urls[i]
            r = requests.get(url)
            file_name = f"{word}_{i}.mp4"
            file_path = os.path.join(word_folder, file_name)
            open(file_path, "wb").write(r.content)


if __name__ == "__main__":
    """Downloads signbsl.com videos from the url file produced

    Returns
    ---------
    A folder containing all of the downloaded videos
    """
    output_folder = "../new_data/"

    with open("signbsl_video_url.txt", "r") as f:
        lines = f.readlines()

    args = ((line, output_folder) for line in lines)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        executor.map(download, lines, repeat(output_folder))
