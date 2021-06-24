from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from time import sleep
import string

if __name__ == "__main__":
    """Scrapes all the words from A-Z from the signbsl.com website

    Returns
    ---------
    Saves the url of each letter to a txt
    """

    url = "https://www.signbsl.com/dictionary/"
    driver = webdriver.Chrome()
    driver.set_window_size(1920, 1080)

    data = []

    for letter in string.ascii_lowercase:
        current_url = url + letter
        driver.get(current_url)
        result_pages = len(driver.find_elements_by_class_name("pagination")[1].find_elements_by_tag_name("li"))

        for i in range(1, result_pages + 1):
            driver.get(current_url + "/" + str(i))
            words = [
                elem.get_attribute("href")
                for elem in driver.find_element_by_tag_name("table").find_elements_by_tag_name("a")
            ]
            data.extend(words)

    with open("signbsl_word_list.txt", "w") as f:
        for line in data:
            f.write(line)
            f.write("\n")

    driver.close()
