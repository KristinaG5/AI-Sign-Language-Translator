import argparse
import os
import subprocess


def get_length(input_video):
    result = subprocess.run(
        ["ffprobe", "-i", input_video, "-show_entries", "format=duration", "-v", "quiet", "-of", "csv='p=0'"]
    )


def run_ffmpeg_command(input_path, start_time, end_time, output):
    subprocess.run(["ffmpeg", "-ss", start_time, "-i", input_path, "-to", end_time, "-c", "copy", output])
    # subprocess.run(["ffmpeg", "-i", input_path, "-ss", start_time, "-t", duration, output], capture_output=True)


def trim_videos(folder_path):
    video_files = os.listdir(folder_path)
    for video in video_files:
        get_length(video)
        print(video)
    # ffmpeg_extract_subclip("video1.mp4", start_time, end_time, targetname="test.mp4")


if __name__ == "__main__":
    """
    TODO: implement this function later
    Trims the start and end of word videos to get the main action bit.
    """

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-f", "--folder_path", help="Path to folder where the video frames are located", type=str, required=True
    )
    args = parser.parse_args()

    trim_videos(args.folder_path)
