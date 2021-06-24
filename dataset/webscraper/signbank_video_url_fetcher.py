from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from time import sleep
import string

if __name__ == "__main__":
    """Saves the url for each individual word video from the csv produced from the BSL Signbank website

    Returns
    ---------
    A txt file containing all of the individual word videos
    """

    driver = webdriver.Chrome()
    driver.set_window_size(1920, 1080)

    data = set()

    with open("word_urls.txt", "r") as f:
        urls = f.readlines()

    for url in urls:
        driver.get(url)
        results = len(driver.find_element_by_xpath('//*[@id="signinfo"]/div[2]/div').find_elements_by_tag_name("a")) + 1
        driver.switch_to.frame(driver.find_element_by_id("videoiframe"))
        src = driver.find_element_by_tag_name("video").get_attribute("src")
        data.add(src)

        for i in range(2, results + 1):
            driver.get(url.replace("1", str(i)))
            driver.switch_to.frame(driver.find_element_by_id("videoiframe"))
            src = driver.find_element_by_tag_name("video").get_attribute("src")
            data.add(src)

    with open("word_video_url.txt", "w") as f:
        for line in data:
            f.write(line)
            f.write("\n")

    driver.close()
