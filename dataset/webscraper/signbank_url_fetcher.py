from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from time import sleep
import string

if __name__ == "__main__":
    """Scrapes all the words from A-Z from the BSL Signbank website

    Returns
    ---------
    Saves the url of each letter to a txt
    """

    url = "https://bslsignbank.ucl.ac.uk/dictionary/search/?"
    driver = webdriver.Chrome()
    driver.set_window_size(1920, 1080)

    data = []

    for letter in string.ascii_uppercase:
        current_url = url + "query=" + letter
        driver.get(current_url)
        try:
            result_pages = (
                len(driver.find_element_by_xpath('//*[@id="wrap"]/div/div[5]/p[3]').find_elements_by_tag_name("a")) + 1
            )
        except NoSuchElementException:
            result_pages = 1

        for i in range(1, result_pages + 1):
            driver.get(current_url + "&page=" + str(i))
            table = driver.find_element_by_id("searchresults")
            links = table.find_elements_by_tag_name("a")
            for link in links:
                data.append(link.get_attribute("href"))

    with open("word_urls.txt", "w") as f:
        for line in data:
            f.write(line)
            f.write("\n")

    driver.close()
