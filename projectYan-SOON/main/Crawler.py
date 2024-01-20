from selenium import webdriver
from selenium.webdriver.common.by import By
import re
import time

def crawl_yanchan():
    driver = webdriver.Chrome()

    driver.get(url = "https://arca.live/b/yandere?category=%EC%8D%A8%EC%A4%98")


    for page in range(1, 11):
        pageClicker = driver.find_element(By.XPATH, f'/html/body/div[2]/div[3]/article/div/nav/ul/li[{page}]/a')
        pageClicker.click()
        time.sleep(1)

        for i in range(45):
            k = i + 9
            novelLinks = driver.find_element(By.XPATH, f'/html/body/div[2]/div[3]/article/div/div[6]/div[2]/a[{k}]/div')
            novelLinks.click()
            time.sleep(1)

            title = driver.find_element(By.XPATH, '/html/body/div[2]/div[3]/article/div/div[2]/div[2]/div[1]/div/text()').text
            totText = driver.find_element(By.XPATH, "/html/body/div[2]/div[3]/article/div/div[2]/div[4]").text
            line = re.findall('"([^"]*)"', totText)

            with open("projectYan-SOON/data/Data.csv", "a") as f:
                f.write(f"{title},{line}\n")

            driver.back()

    driver.quit()

crawl_yanchan()