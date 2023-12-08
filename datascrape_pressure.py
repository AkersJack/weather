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
import re
import atexit
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
    

def scrape(url):
    

    new_url = None
    def cleanup():
        try:
            print("Cleaning Up")
            if masterdict != {} or len(masterdict) != 0:
                with open("data_pressure.json", 'w') as f:
                    json.dump(masterdict, fp=f, indent=4)
            print(f"Exit URL: {new_url}")
        except Exception as cleanuperror:
            print(f"Cleanup error: {cleanuperror}")
  

    download_directory = "D:\weather"
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

    masterControl = False

    



    
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

    current_cache_size = 0


    # Change the number of processors (number of images to fetch at a time)
    # cache_size = WebDriverWait(driver, 7).until(EC.presence_of_element_located((By.ID, "parallel-proc")))
    # driver.execute_script("""
    #                       arguments[0].setAttribute('data-value','4'); 
    # """, cache_size)

    total_count = 0


    # Pay bypass (Ad block)
    modified_script = """
    var checkChartcounter = function() {
        console.log("Paywall overwrite");
        return 2;
    }
    console.log("Injected"); 
    """
    driver.execute_script(modified_script)

    
    writeCount = 0
    try:
        while total_count < 500: 
            prev_cache_size = current_cache_size 

            try:
                pressure_button = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH,  f'/html/body/div[1]/div/div[6]/div/div/div/div/div[1]/div[1]/div[1]/div[1]/form/div[1]/div[1]/div/div[2]/div/div[2]/div[2]/div[1]/div/a[3]')))
                pressure_button.click()
            except Exception as e: 
                print("error pressure")

            try:
                prev_time = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH,  f'//*[@id="modelselect"]/div[1]/div[2]/div/div[3]/div[2]/span[1]')))
                prev_time.click()
            except Exception as e: 
                driver.execute_script("model_date_prev();")
                time.sleep(1)
                print("Radar offline")
                # new_link = driver.execute_script('return directURL();')
                # print(f"New link: {new_link}")
                # return new_link
        

            # Pay bypass (Ad block)
            driver.execute_script("xclicksvalue = 0;")
            w_count = 0
            while current_cache_size == 0:
                current_cache_size = driver.execute_script("""
                            return document.querySelector("#text-overlay > div.value-container").children.length;
                        """
                        )
                if current_cache_size == prev_cache_size:
                    time.sleep(1)
                    w_count += 1
                if w_count == 3:
                    break

            

            f = open('data_pressure.json', 'r')
            masterdict = json.load(f)
            f.close()
            # print(f"Master dict: {masterdict}")
        
            while current_cache_size > 0: 
                src = driver.execute_script(
                    """
                            return $(document.querySelector("#text-overlay > div.value-container").children[0]).attr("title");
                    """
                )
                src = src.replace("~", "")
                # print(f"Src: {src}")

            
                key = driver.execute_script(
                    """
                    return $('#model-valid').val();
                    """
                )

                # print(f"Key: {key}")


                if key == "null" or key == None:
                    key = False
        
          
                

                # try:
                #     if masterdict[key] == None: 
                #         masterdict[key] = []
                # except Exception as newe: 
                #     masterdict[key] = [src]
                #     print(f"Prob key error: {newe}")
                try:
                    if(key != False):
                        if(key in masterdict):
                            print(f"Adding: Key: {key} Src: {src}")
                            # print(f"Key exists appending: {masterdict[key]}")
                            m = masterdict[key]["pressure"]
                            if src not in (masterdict[key]["pressure"]):
                                masterdict[key]["pressure"].append(src)
                            else: 
                                print("Value already exists in key skipping adding value.")
                            # print(f"new value: {masterdict[key]}")
                        else: 
                            print(f"Key: [{key}] Src: [{src}]")
                            masterdict[key] = {"pressure": [src]} 
                            # dumped = json.dumps(masterdict)
                            # print(f"src after: {src}")
                            # print(f"Masterdict: {masterdict}")
                            # print(f"Dumped dict: {dumped}")
                                

                except Exception as e: 
                    print(f"Exception here: {e}")
                    masterdict[key] = {"pressure":[]}
            
                # formatted = json.dumps(masterdict, indent=4)
                # print(f"formatted: {formatted}")



        
                    #$("#preload-cache-container").children()[0].remove();
            
                # Remove the item from the container
                driver.execute_script(
                    """
                    return document.querySelector("#text-overlay > div.value-container").children[0].remove();
                    """
                                    )
      
                # Update cache size 
                current_cache_size = driver.execute_script("""
                            return document.querySelector("#text-overlay > div.value-container").children.length;
                        """
                        )
            
            # Update every 50 items  
            print(f"Write count: {writeCount}")
            # if writeCount >= 10:
            with open("data_pressure.json", 'w') as newfile:
                print(f"Writing Data")
                json.dump(masterdict, newfile, indent=4)
                writeCount = 0
                # print(f"Writing data: {masterdict}")
                # formatted = json.dumps(masterdict, indent=4)
                # newfile.write(formatted)
            # else:
            #     writeCount += 1


            total_count += 1
            new_link = driver.execute_script('return directURL();')
            # try:
            #     atexit.unregister(cleanup)
            # except Exception as unregisterError: 
                # print(f"No register to unregister")
            # atexit.register(cleanup)
            print(f"Link: {new_link}")
            print(f"Loop count: {total_count}")



    except KeyboardInterrupt:
        print("User forcefully stopped the program.")
        new_link = driver.execute_script('return directURL();')
        print(f"New link: {new_link}")
        cleanup(f, masterdict, new_link)

            
            

    new_link = driver.execute_script('return directURL();')
    print(f"New link: {new_link}")

    print("Final Write")
    with open("data_pressure.json", 'w') as f:
        json.dump(masterdict, fp=f, indent=4)

    time.sleep(3)

        
    return new_link
    
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
    data = "https://img3.weather.us/images/data/cache/radarus/radarus_2022_04_03_2837_KLWX_357_104754.png"
    # new_url = "https://weather.us/observations/baltimore/precipitation-total-1h-in100/20231103-1200z.html"
    # new_url = "https://weather.us/observations/baltimore/pressure-qff/20231130-0300z.html"
    # new_url = "https://weather.us/observations/baltimore/pressure-qff/20230722-0800z.html"
    # new_url = "https://weather.us/observations/baltimore/pressure-qff/20230722-0600z.html"
    # new_url =  "https://weather.us/observations/baltimore/pressure-qff/20230717-0300z.html"
    #new_url = "https://weather.us/observations/baltimore/pressure-qff/20230430-0500z.html"
    # new_url = "https://img1.weather.us/images/data/cache/radarus/radarus_2023_04_20_2837_KLWX_357_054053.png"
    # new_url = "https://weather.us/observations/baltimore/pressure-qff/20221122-0700z.html"
    # new_url = "https://weather.us/observations/baltimore/pressure-qff/20220824-0600z.html"
    # new_url = "https://weather.us/observations/baltimore/pressure-qff/20231130-0500z.html"
    new_url = "https://weather.us/observations/baltimore/pressure-qff/20220314-0400z.html"

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

    for x in range(duration):
        print(f"Progress: {x}/{duration}")
        print(f"New URL: {new_url}")
        # time.sleep(random.randint(0, 10))
        new_url = scrape(new_url)
        
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