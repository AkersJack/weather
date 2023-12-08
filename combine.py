import base64
from PIL import Image
import cv2
import json
import re
import datetime
import os 
import time



# precip_data_f = open("data.json", "r")
# precip_data = json.load(precip_data_f)
# precip_data_f.close()

# pressure_data_f = open("data_pressure.json", "r")
# pressure_data = json.load(pressure_data_f)
# pressure_data_f.close()

# dewpoint_data_f = open("data_dewpoint.json", "r")
# dewpoint_data = json.load(dewpoint_data_f)
# dewpoint_data_f.close()

dps_data_f = open("data_dpspread.json", "r")
dps_data = json.load(dps_data_f)
dps_data_f.close()

humidity_data_f = open("data_humidity.json", "r")
humidity_data = json.load(humidity_data_f)
humidity_data_f.close()

temp_data_file = open("data_temp.json", "r")
temp_data = json.load(temp_data_file)
temp_data_file.close()

wb_data_file = open("data_wb.json", "r")
wb_data = json.load(wb_data_file)
wb_data_file.close()

file = open("./data_new.json", 'r')
master_data = json.load(file)
file.close()



def parseKey(key): 
    regex = r"(\d{4})-(\d{2})-(\d{2})/(\d{2}):(\d{2})"
    match = re.search(regex, key)
    if match:
        date = match.group(1) + "-" + match.group(2) + "-" + match.group(3)
        time = match.group(4) + ":" + match.group(5)
        # print(f"Date: {date}")
        # print(f"Time: {time}")
        date_time = date + " " + time
        dt = datetime.datetime.strptime(date_time, "%Y-%m-%d %H:%M")
    else:
        print(f"Error: No match found: {key}")
    
    return(dt)
    

# Used to parse the images file names to get the actual date time so they can be matched with the precip data 
def parseFile(filename):
    regex = r"(\d{4})-(\d{2})-(\d{2})\.(\d{2})_(\d{2})_(\d{2})\.png"
    match = re.search(regex, filename)
    if match:
        year = match.group(1)
        month = match.group(2)
        day = match.group(3)
        hour = match.group(4)
        minute = match.group(5)
        second = match.group(6)
        # print("Year:", year)
        # print("Month:", month)
        # print("Day:", day)
        # print("Hour:", hour)
        # print("Minute:", minute)
        # print("Second:", second)
        date_time = f"{year}-{month}-{day} {hour}:{minute}:{second}" 
        dt = datetime.datetime.strptime(date_time, "%Y-%m-%d %H:%M:%S")
        # print(dt)
    else: 
        print("Error: No match found: {filename}")
    return dt


if __name__ == "__main__": 
    # Merging all the data to 1 file was manually done for each of the commented items 


    # master_data = {}
    
    # merge_item = 
    # key_label = 


    # merge_item = dewpoint_data
    # key_label = "dewpoint"

    # merge_item = dps_data
    # key_label = "dewpoint_spread"

    # merge_item = humidity_data
    # key_label = "humidity"

    # merge_item = temp_data
    # key_label = "temperature"

    
    # merge_item = wb_data
    # key_label = "wetbulbtemp"


    # merge_item = pressure_data
    # key_label = "pressure"

    # merge_item = precip_data
    # key_label = "precipitation"

    
    def combineFiles(merge_item, key_label):
        print("Combining...")
        for key in merge_item: 
            if key in master_data:
                # print(f"Merge item key {merge_item[key]}")
                master_data[key].update(merge_item[key])
                # print(f"Merged: {master_data[key]}")
            else: 
                master_data[key] = merge_item[key]
            
        with open("./data_new.json", "w") as f: 
            # master_data.update(precip_data)
            json.dump(master_data, f, indent = 4)

        print("Done!")
    # Parse the keys into date and times 

    # for key in master_data: 
    #     parseKey(key)


    # Image Combine

    # This takes a lot of time to run because it goes through all 100,000 images each time to find the an image whose date and time most closely matches the precip date and time. 
    # Dynamic programming would drastically increase the speed 

    pictures = os.listdir("./weather")
    # for picture in pictures: 
        # ptime = parseFile(picture)
        
    possible_pictures = 0
    better_pictures = 0
    amazing_pictures = 0
    count = 0
    key_list = list(master_data.keys())
    for key in key_list:
        best_possible = None
        best_possible_picture = None
        print(f"Count: {count}/{len(key_list)}") 
        if count % 100 == 0:
            print(f"Total Same Day: {possible_pictures}") 
            print(f"Total Same Hour: {better_pictures}") 
            print(f"Total Within 30 min: {amazing_pictures}") 
        k_time = parseKey(key)
        count += 1
        for picture in pictures:
            ptime = parseFile(picture)
            time_delta = abs(ptime - k_time)
            # print(f"Difference: {time_delta.seconds}")
            # time.sleep(1)
            if(time_delta.days == 0):
                # print(f"Same day: {time_delta.seconds}")
                # print(f"Master Dict: {k_time}")
                # print(f"Picture: {ptime}")
                
                # loaded_image = Image.open(os.path.join("./weather/", picture))
                # width, height = loaded_image.size
                
                # if width == 760 and height == 616: 
                if best_possible_picture == None or best_possible == None:
                    best_possible_picture = picture
                    best_possible = time_delta

                elif(best_possible.seconds > time_delta.seconds):
                    best_possible_picture = picture
                    best_possible = time_delta

                possible_pictures += 1
                # else:
                #     print(f"Bad image: {picture}")
                    

                # possible_pictures.append(time_delta)
                if(time_delta.seconds <= 3600):
                    # better_pictures.append(time_delta.seconds)
                    better_pictures += 1
                if(time_delta.seconds <= 1800):
                    # amazing_pictures.append(time_delta.seconds)
                    amazing_pictures += 1
                    
                # time.sleep(1)
                 
        
        if "picture" not in master_data[key]:
            # print("New Key")
            master_data[key]["picture"] = best_possible_picture
            if count % 1000 == 0: 
                with open("./data_new.json", "w") as f: 
                    json.dump(master_data, f, indent = 4) 
        else:
            print("Key already exists")
            pass
                
    with open("./data_new.json", "w") as f: 
        json.dump(master_data, f, indent = 4) 

    
    print(f"Total Same Day: {possible_pictures}") 
    print(f"Total Same Hour: {better_pictures}") 
    print(f"Total Within 30 min: {amazing_pictures}") 
    
    print("Done!")
        
        
        
