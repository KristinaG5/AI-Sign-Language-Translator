from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from time import sleep
import string
import os

if __name__ == "__main__":
    """Saves the url for each individual word video from the csv produced from the signbsl.com website

    Returns
    ---------
    A txt file containing all of the individual word videos
    """

    driver = webdriver.Chrome()
    driver.set_window_size(1920, 1080)

    data = set()

    with open("signbsl_word_list.txt", "r") as f:
        urls = f.readlines()

    videos = {}
    for url in urls:
        driver.get(url)
        word = os.path.basename(url).strip("\n")
        print(word)
        src = [
            elem.get_attribute("content") for elem in driver.find_elements_by_xpath("//meta[@itemprop='contentURL']")
        ]
        videos[word] = src

    with open("signbsl_video_url.txt", "w") as f:
        for word, url in videos.items():
            f.write(f"{word}|{url}\n")

    driver.close()
