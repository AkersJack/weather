from bs4 import BeautifulSoup as bs
from selenium.webdriver.support import expected_conditions as EC 
from selenium.webdriver.support.ui import WebDriverWait
from selenium import webdriver
from selenium.webdriver.common.by import By
import time 
import requests 
import pickle
from selenium.webdriver.common.action_chains import ActionChains
import requests
import json
import random
from io import BytesIO
import re 
# Not headless 

# Website map: https://weather.us/radar-us/baltimore/reflectivity/KLWX_20170812-050638z.html
# Overlay image: 


url = "https://weather.us/radar-us/baltimore/reflectivity/KLWX_20231123-060030z.html"
# url = "https://weather.us/radar-us/baltimore/reflectivity/KLWX_20231122-060131z.html"
# url = "https://weather.us/radar-us/baltimore/reflectivity/KLWX_20231121-060145z.html"
# url = "https://weather.us/radar-us/baltimore/reflectivity/KLWX_20180515-050509z.html"
# url = "https://weather.us/radar-us/baltimore/reflectivity/KLWX_20170901-050045z.html"

# url = "https://weather.us/radar-us/baltimore/reflectivity/KLWX_20170814-050825z.html"


# id = preload-cache-container
def linkify(date, t):
    hour = t[0:2]
    minute = t[3:5]
    second = t[6:8]
    am_pm = t[8:10]

    day = date[0:2]
    month = date[3:5]
    year = date[6:10]
    

    hour = int(hour) + 5

    base = "https://img3.weather.us/images/data/cache/radarus/radarus_"

    


    if am_pm == "pm":
        hour += 12
        hour = hour % 12
    
    formatted_url = base +  year + "_" + month + "_" + day + "_" + "2837" + "_" + "KLWX" + "_" + "357" + "_" + str(hour) + str(minute) + str(second) + ".png"
    
    return formatted_url
    


def scrape(url, start):
    cr_image = open("badimage.png", "rb")
    no_image = open("noimage.png", "rb")

    cr_image_b = cr_image.read()
    no_image_b = no_image.read()

  

    download_directory = f"D:/weather/"
    chrome_options = webdriver.ChromeOptions()
    prefs = {"download.default_directory": download_directory}
    chrome_options.add_experimental_option("prefs", prefs)
    chrome_options.add_argument("--headless=new")

    driver = webdriver.Chrome(options=chrome_options)

    driver.refresh()


    # driver.maximize_window()

    driver.get(url)

    
    script =  WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "/html/body/script[10]")))
    # print(f"Script URL: {script.src}")


    


    
    # Get past the terms and conditions (if they pop up)
    element = WebDriverWait(driver, 7).until(EC.presence_of_element_located((By.ID, "sp_message_iframe_918538")))
    switchTo = driver.switch_to 
    switchTo.frame(element); 

    button = WebDriverWait(driver, 7).until(EC.presence_of_element_located((By.XPATH, f'//*[@id="notice"]/div[3]/div[1]/button')))
    button.click()
    
    switchTo.default_content()
    time.sleep(1)

    

    # Change the number of elements to cache in the html file 
    # cache_size = WebDriverWait(driver, 7).until(EC.presence_of_element_located((By.ID, "preload-count")))
    # cache_size = driver.find_element(By.ID, "preload-count")
    # cache_size.set_attribute('data-value', "200")
    # It will only do 295 max
    # driver.execute_script("""
    #                       arguments[0].setAttribute('data-value','0'); 
    # """, cache_size)


    # Change the number of processors (number of images to fetch at a time)
    # cache_size = WebDriverWait(driver, 7).until(EC.presence_of_element_located((By.ID, "parallel-proc")))
    # driver.execute_script("""
    #                       arguments[0].setAttribute('data-value','4'); 
    # """, cache_size)

    total_count = 0

    def parseDateTime(url):
        v = url.index("_")
        url = url[v:]
        pattern = r"_(\d{4})_(\d{2})_(\d{2})_(\d{4})_(\w{4})_(\d{3})_(\d{6}).png"
        matched = re.match(pattern, url)
        year = matched[1]
        month = matched[2]
        day = matched[3]
        time = matched[7]
        formatted_time = time[:2] + "_" + time[2:4] + "_" + time[4:6]
        formatted_dt = year + "-" + month + "-" + day + "." + formatted_time + ".png"
        return formatted_dt



        



    # Pay bypass (Ad block)
    modified_script = """
    var checkChartcounter = function() {
        console.log("Paywall overwrite");
        return 2;
    }
    console.log("Injected"); 
    """
    driver.execute_script(modified_script)

    

    file = open("links", 'r')
    src = file.readline()
    src = src.strip('\n')

    while(src != start):
        src = file.readline() 
        src = src.strip('\n')
    
    bad_image = 0
    while total_count < 500: 
        driver.execute_script("xclicksvalue = 0;")

        driver.get(src)
        image_element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "/html/body/img")))
        image_url = driver.execute_script("""
            return document.querySelector("body > img").src;
        """
        )
        print(f"Image URL: {image_url}")
        response = requests.get(image_url)
        image_data = BytesIO(response.content)
        name = parseDateTime(image_url)
        image_data_r = image_data.read()
        if image_data_r != cr_image_b and image_data_r != no_image_b:            
            bad_image = 0
            with open(download_directory + name,"wb") as f: 
                print(f"Image saved: {download_directory + name}")
                f.write(image_data_r)
            total_count += 1
            print(f"Image number: {total_count}")

            src = file.readline()
        else:
            bad_image += 1
            print("Bad Image Avoided")
            src = file.readline()
            if bad_image == 10 :
                if image_data == no_image_b:
                    print("No image")
                return src
    
    
    src = file.readline()



    cr_image.close()
    file.close()

    return src 


    
    # get the next link 

#preload_urls 
# 295 children per pae (295 times)
# model_valids_to_preload (not public )
# preload_urls

# get_selected_
# $('#model-valid').val(); // Get the date and time (current)
# get_selected_model_path() // [date(with underscores), time]
# model_hour_prev()
# model_date_prev

# mid = "#model-run" or "#model-valid"  model-run for date model-valid for time
#   var items = model_valids_get_item_count(mid+" option"); -> returns number of valid items

# Insert this 
    """
    var checkChartcounter = function() {
        return true;
}
    """

def downloadImages(link):
    base_url = url = "https://weather.us/radar-us/baltimore/reflectivity/KLWX_20231123-060030z.html"
    download_directory = "D:\weather"
    chrome_options = webdriver.ChromeOptions()
    prefs = {"download.default_directory": download_directory}
    chrome_options.add_experimental_option("prefs", prefs)

    driver = webdriver.Chrome(options=chrome_options)

    driver.refresh()

    # driver.maximize_window()

    driver.get(url)

    
    # Get past the terms and conditions (if they pop up)
    element = WebDriverWait(driver, 7).until(EC.presence_of_element_located((By.ID, "sp_message_iframe_918538")))
    switchTo = driver.switch_to 
    switchTo.frame(element); 

    button = WebDriverWait(driver, 7).until(EC.presence_of_element_located((By.XPATH, f'//*[@id="notice"]/div[3]/div[1]/button')))
    button.click()
    
    switchTo.default_content()
    time.sleep(3)
    
    


    
    


    
    




#https://img1.weather.us/images/data/cache/radarus/radarus_2023_11_03_2837_KLWX_357_005424.png

# download = "https://weather.us/radar-us/baltimore/reflectivity/KLWX_20170812-050638z.html"
# overlay = "https://img3.weather.us/images/data/cache/radarus/radarus_2017_08_12_2837_KLWX_357_050638.png"
# save_as()
#                refreshShareURL();
# directURL();
# Year-month-day
# 7 hours behind the actual date is the URL date 
# 11 - 4
# 10 - 3
# 5 - 22
# 7 - 0
# URL 
# //*[@id="popover430930"]/div[2]/div[1]/div[2]/input 

"""
    https://weather.us/radar-us/baltimore/reflectivity/KLWX_20231103-005424z.html 
    
    https://weather.us/radar-us/baltgary/reflectivity/
    
    KLWX_
    
    yearmonthday
    -
    (hour - 7)minutesecond
    z.html

    
"""
#id = preload-count
#id = preload-cache-container
    

def writeFile(data):
    with open('links', 'a') as outfile:
        outfile.write(data)



    
    # time_text = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH,  '//select[@id="model-valid"]//option[@selected="selected"]'))) # time_text = time_text.text     



# Weather observations 
# https://weather.us/observations/baltimore/weather-observation/20231129-1500z.html 

# Model param 210 temp


# https://img1.weather.us/images/data/cache/radarus/radarus_2023_11_22_2837_KLWX_357_060131.png
# 2023_11_22_2837_060131.
# download = "https://weather.us/radar-us/baltimore/reflectivity/KLWX_20170812-050638z.html"
# download = "https://weather.us/radar-us/baltimore/reflectivity/KLWX_20231122-073357z.html"

if __name__ == '__main__': 
    # data = "https://img1.weather.us/images/data/cache/radarus/radarus_2023_05_17_2837_KLWX_357_164800.png"
    # data = "https://img4.weather.us/images/data/cache/radarus/radarus_2023_05_02_2837_KLWX_357_151701.png"
    # data = "https://img3.weather.us/images/data/cache/radarus/radarus_2023_04_12_2837_KLWX_357_104159.png"
    # data = "https://img1.weather.us/images/data/cache/radarus/radarus_2022_06_20_2837_KLWX_357_081436.png"
    # data = "https://img1.weather.us/images/data/cache/radarus/radarus_2022_04_04_2837_KLWX_357_061219.png"
    data = "https://img3.weather.us/images/data/cache/radarus/radarus_2022_04_03_2837_KLWX_357_104754.png"
    # new_url = "https://weather.us/radar-us/baltimore/reflectivity/KLWX_20220403-010403z.html"
    # new_url = "https://img3.weather.us/images/data/cache/radarus/radarus_2022_03_26_2837_KLWX_357_025007.png"
    new_url = "https://weather.us/radar-us/baltimore/reflectivity/KLWX_20211024-024203z.html"
    # start = "https://img1.weather.us/images/data/cache/radarus/radarus_2023_11_21_2837_KLWX_357_075849.png"
    # start = "https://img4.weather.us/images/data/cache/radarus/radarus_2023_11_12_2837_KLWX_357_120314.png"
    # start = "https://img3.weather.us/images/data/cache/radarus/radarus_2023_10_18_2837_KLWX_357_114746.png"
    # start = "https://img3.weather.us/images/data/cache/radarus/radarus_2023_10_10_2837_KLWX_357_134623.png"
    start = "https://img1.weather.us/images/data/cache/radarus/radarus_2023_06_29_2837_KLWX_357_154257.png"
    # start = "https://img1.weather.us/images/data/cache/radarus/radarus_2023_06_29_2837_KLWX_357_154710.png"
    # start = "https://img1.weather.us/images/data/cache/radarus/radarus_2023_06_29_2837_KLWX_357_154257.png"
    # start = "https://img3.weather.us/images/data/cache/radarus/radarus_2023_06_08_2837_KLWX_357_063931.png"
    # start = "https://img3.weather.us/images/data/cache/radarus/radarus_2022_06_01_2837_KLWX_357_165048.png"
    # start = "https://img3.weather.us/images/data/cache/radarus/radarus_2022_06_01_2837_KLWX_357_165048.png"
    start = "https://img4.weather.us/images/data/cache/radarus/radarus_2021_11_10_2837_KLWX_357_063129.png"


    # new_url = scrape(url)
    # print(f"URL: {new_url}")
    duration = 3650
    with open('links', 'r') as f: 
        # data = f.readline()
        data = data.strip('\n')
        print(f"Line: {data}")
        # data = data.replace("https://img1.weather.us/images/data/cache/radarus/radarus_", "")

        base = "https://weather.us/radar-us/baltimore/reflectivity/KLWX_"
        data = data[58:-3]
        data = data.replace("_357_", "")
        data = data.replace("_2837", "")
        data = data.replace("KLWX", "")
        data = list(data) 
        data[-8] = '-'
        data = ''.join(data)
        data = data.replace(".", "")
        data = data.replace('_', "")

            
        # new_url = base + data + "z.html"

    new_url = "https://weather.us/radar-us/baltimore/reflectivity/KLWX_20211024-024203z.html"
    for x in range(duration):
        print(f"Progress: {x}/{duration}")
        print(f"New URL: {new_url}")
        # time.sleep(random.randint(0, 10))
        start = scrape(new_url, start)
        start = start.strip('\n')
        print(f"Start: {start}")
        
    #hc_obs_series
    #History.getBasePageUrl()
    
    
    
    
    """
    Station: KDMH
    Dewpoint 
    Temperature
    relative humidity
    Sea level pressure QFF
    Sea level pressure QHN
    dew point (degrees F)
    dew point spread  (degrees F)
    wet bulb temperature (degrees F)
    precipitation total, 1h (1/100 in)
    snow depth (in) 
    """