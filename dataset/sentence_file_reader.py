import xml.etree.ElementTree as ET
from datetime import datetime
import subprocess
from tqdm import tqdm

if __name__ == "__main__":
    """Splits the conversation videos into individual sentences based on the eaf file

    Returns
    ---------
    A folder containing the sentence corresponding to the video time slot to a txt file
    """

    name = "G24n"
    tree = ET.parse(f"dataset/eafs/{name}.eaf")
    root = tree.getroot()

    # get start time
    for header in root.iter("MEDIA_DESCRIPTOR"):
        start_time = int(header.attrib["TIME_ORIGIN"])
        break

    # get time slots
    times = {time.attrib["TIME_SLOT_ID"]: int(time.attrib["TIME_VALUE"]) for time in root.iter("TIME_SLOT")}

    def run_ffmpeg_command(input_path, start_time, duration, output):
        subprocess.run(["ffmpeg", "-i", input_path, "-ss", start_time, "-t", duration, output], capture_output=True)

    labels = {}
    # get sentences
    tier = root.find('./TIER[@TIER_ID="Free Translation"]')
    for child in tqdm(tier.iter("ALIGNABLE_ANNOTATION")):
        time_slot_start = child.attrib["TIME_SLOT_REF1"]
        time_slot_end = child.attrib["TIME_SLOT_REF2"]
        text = child[0].text
        start = times[time_slot_start] + start_time
        end = times[time_slot_end] + start_time + 300

        duration = (end - start) / 1000
        filename = f"{name}_{start}.mov"
        run_ffmpeg_command(
            f"data/val_videos/{name}/{name}.mov",
            str(start / 1000),
            str(duration),
            f"data/val_videos/{name}/" + filename,
        )
        labels[filename] = text

    with open(f"data/val_videos/{name}/{name}_data.txt", "w") as f:
        for filename, text in labels.items():
            f.write(f"{filename}|{text}\n")
