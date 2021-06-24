from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from time import sleep

if __name__ == "__main__":
    """Scrapes a list of sentence video urls from the BSL Corpus website

    Returns
    ---------
    Saves urls to a csv file
    """

    driver = webdriver.Chrome()
    driver.set_window_size(1920, 1080)
    driver.get("http://digital-collections.ucl.ac.uk/R/?local_base=BSLCP")

    driver.find_element_by_xpath('//*[text()="Browse by activity"]').click()
    driver.find_element_by_xpath('//*[text()="Narrative"]').click()
    driver.find_element_by_xpath('//*[text()="Full view"]').click()

    data = []

    while True:
        table = driver.find_element_by_xpath('//*[@id="ucl_container"]/table[2]/tbody/tr/td/table/tbody/tr[3]/td/table')
        rows = table.find_elements_by_tag_name("tr")

        available_data = rows[1].text.split("\n")[1:]
        i = 0
        erf_index = None
        mov_index = None
        for row in available_data:
            if "EAF" in row:
                erf_index = i
            elif "Quicktime" in row:
                mov_index = i
            i += 1

        print(f"ERF: {erf_index}, MOV: {mov_index}")
        if erf_index is not None and mov_index is not None:
            urls = [row.text for row in rows if row.text.startswith("URI")]
            erf_url = urls[erf_index].split(" ")[1]
            mov_url = urls[mov_index].split(" ")[1]
            data.append((erf_url, mov_url))
            print(len(data))

        try:
            driver.find_element_by_xpath('//*[@title="Next Page"]').click()
        except:
            break

    with open("sentence_urls.csv", "w") as f:
        f.write("erf,mov\n")
        for erf, mov in data:
            f.write(erf)
            f.write(",")
            f.write(mov)
            f.write("\n")

    driver.close()
