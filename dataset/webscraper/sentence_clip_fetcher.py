from selenium import webdriver
import os
import time
import argparse


def file_is_downloaded():
    """Check if video is being downloaded

    Returns
    ---------
    True
    """
    time.sleep(3)
    downloaded = False

    while not downloaded:
        files = os.listdir(args.location)
        downloaded = True
        for file in files:
            if ".crdownload" in file:
                downloaded = False
                print("STILL DOWNLOADING")
                time.sleep(1)

    return True


if __name__ == "__main__":
    """Downloads videos from the sentence_urls.csv list

    Returns
    ---------
    Saves videos to downloads path
    """

    # Arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-l", "--location", help="Location to where the videos will be downloaded to", type=str, required=True
    )
    args = parser.parse_args()

    with open("sentence_urls.csv") as f:
        data = f.readlines()[1:]

    urls = []
    for item in data:
        erf, mov = item.split(",")
        urls.append(erf)
        urls.append(mov)

    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    prefs = {"download.default_directory": args.location}
    options.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome(options=options)
    i = 0

    for url in urls:
        driver.get(url)
        driver.switch_to.frame(driver.find_element_by_name("MainFrame"))
        driver.switch_to.frame(driver.find_element_by_name("SideMain"))
        driver.find_element_by_xpath('//*[text()="Continue >>"]').click()

        assert file_is_downloaded()
        print(i)
        i += 1

    driver.close()
