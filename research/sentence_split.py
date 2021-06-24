import xml.etree.ElementTree as ET
import re
import os

if __name__ == "__main__":
    """Finds which sentence videos contains the most amount of words based on the word dataset produced

    Returns
    ---------
    Prints the information to console
    """
    regex = re.compile("[^a-z ]")
    available_words = set(os.listdir("data/trimmed_dataset_5_val"))

    for filename in os.listdir("dataset/eafs"):
        tree = ET.parse(os.path.join("dataset/eafs", filename))
        root = tree.getroot()

        tier = root.find('./TIER[@TIER_ID="Free Translation"]')
        for child in tier.iter("ALIGNABLE_ANNOTATION"):
            text = child[0].text
            if text:
                text = regex.sub("", text.lower().strip())
                words = [t for t in text.split(" ") if t]
                missing_words = set(words) - available_words
                if not missing_words or len(words) > 6 and len(missing_words) < 3:
                    print(filename, child[0].text, missing_words)
