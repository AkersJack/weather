import json 
import csv 
import re
import time 
from PIL import Image
import pickle 
import os
import numpy as np
import pandas as pd
import base64
import shutil
from sklearn.impute import KNNImputer


# Checks to ensure that all images are radar images and not copyright or error images
def check_images(): 
    image_directory = "./weather"
    pictures = os.listdir(image_directory) # image directory 
    counter = 0
    for picture in pictures:
        if counter % 100 == 0: 
            print(f"{counter}/{len(pictures)}")
        image_path = os.path.join(image_directory, picture) # image 
        image = Image.open(image_path) 
        width, height = image.size

        # check to make sure it is an actual radar image and not an error image
        if width != 760 or height != 616: 
            print(f"Bad size: {picture} ")
            time.sleep(10)
        counter += 1

# Parse the text for the data (parses out the amount of rain and where it happened)
def parsetext(text):
    regex = r"^(.*?)\|(.*?)\|(.*?)$"
    match = re.match(regex, text)
    if match: 
        amount = match.group(1)
        location = match.group(2)
        time = match.group(3)
        # print(f"Amount: {amount}")
        # print(f"Location: {location}")

        regex2 = r"^(.*?)/(.*?)$"
        match2 = re.search(regex2, amount)
        amount_format = float(match2.group(1))

        regex3 = r"^(.*?) \(.*?\)$"
        location = location.strip()
        match3 = re.search(regex3, location)
        try:
            location_format = match3.group(1)
        except Exception as matchError: 
            print(f"No match")
        # print(f"Amount: {amount_format}")


    else:
        print("No match found")
    return amount_format, location_format


# Clean up the data by combining images with precipitation into 1 csv file by date 
def cleanup():
    labels = ["Carroll County", "Baltimore MD", "Gaithersburg", "Baltimore-Martin MD"]

    csv_f = "./Data.csv"
    image_dict = {}

    with open("data_new.json", "r") as f: 
        data = json.load(f)
    
    print("Cleaning up...")
        
    with open(csv_f, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["CarrollCounty", "Baltimore", "Gaithersburg", "Baltimore-Martin","NoRain", "Image"])
        counter = 0
        for x in data: 
            print(f"Counter: {counter}")
            try:
                if data[x]["picture"] != None:
                    try: 
                        
                        # No rain most likely if it didn't record any precipitation
                        write_list = [0, 0, 0, 0, 0, data[x]["picture"]]

                        # Move pictures from the general weather directory (where all scraped images were stored)
                        # to weather_used (which is only images that are used in the dataset)
                        source = os.path.join("./weather/", data[x]["picture"])
                        destination = os.path.join("./weather_used/", data[x]["picture"])

                        image_dict[data[x]["picture"]] = destination

                        if "precipitation" not in data[x]:
                            # writer.writerow(write_list)
                            # counter += 1
                            pass
                        else: 
                            # copy the image to the new location 
                            shutil.copy(source, destination)
                            for i in data[x]["precipitation"]:
                                amount, location = parsetext(i)
                                location = location.strip()
                                # print(f"Location: {location} Amount: {amount}")

                                if amount > 0: 
                                    amount = int(1)
                                else: 
                                    amount = int(0)
                                if location == labels[0]:
                                    write_list[0] = amount
                                elif location == labels[1]:
                                    write_list[1] = amount
                                elif location == labels[2]:
                                    write_list[2] = amount
                                elif location == labels[3]: 
                                    write_list[3] = amount
                                else:
                                    print(f"Error unknown location: {location}")
                            # print(write_list)
                            counter += 1
                            if write_list[0] == 0 and write_list[1] == 0 and write_list[2] == 0 and write_list[3] == 0:
                                write_list[4] = 1
                            writer.writerow(write_list)

                        # time.sleep(3)
 
                
                    except Exception as e: 
                        print(f"Exception: {e}")
                        print(f"Data: {data[x]}")
                        # time.sleep(5)
                        pass
            except Exception as enopic:
                print(f"No picture available skipping.")


    with open("Images.json", "w") as imageFile:
        json.dump(image_dict, imageFile, indent = 4)
        

    print(f"Counter: {counter}")

# used to parse the measurement and the location out of the json file of collected data 
def parseTree(text):
    regex = r"([0-9,]+[\.0-9]*)\s*\|\s*(.*?)\s*\("
    match = re.search(regex, text)
    if match: 
        amount = match.group(1)
        location = match.group(2)
        # print(f"Amount: {amount}")
        # print(f"Location: {location}")
    return amount, location
     

# used to combine all the json data into 1 file 
# this was manually done for each item  
def cleanupTree(): 
    labels_precip = ["Carroll County", "Baltimore MD", "Gaithersburg", "Baltimore-Martin MD"]
    # labels_pressure = ["FSNM2", "Baltimore MD", "Baltimore-Martin MD", "Gaithersburg", "TCBM2", "BUOY", "SHIP", "Patapsco", "BLTM2", "FSKM2"]
    labels_pressure = ["FSNM2", "Baltimore MD", "Gaithersburg", "SHIP", "FSKM2", "BLTM2", "TCBM2", "Baltimore-Martin MD"]
    # labels_dewpoint = ["Carroll County", "Baltimore-Martin MD", "Gaithersburg", "Baltimore MD"]
    labels_dewpoint = ['Carroll County', 'Baltimore-Martin MD', 'Gaithersburg', 'Baltimore MD', 'SHIP'] 
    # labels_dewpoint_spread = ["Gaithersburg","Baltimore MD","Carroll County","Baltimore-Martin MD"]
    labels_dewpoint_spread = ['Gaithersburg', 'Baltimore MD', 'Carroll County', 'Baltimore-Martin MD', 'SHIP']
    # labels_wetbulbtemp = ["Gaithersburg","Baltimore MD","Carroll County","Baltimore-Martin MD", "SHIP"]
    labels_wetbulbtemp = ['Gaithersburg', 'Baltimore-Martin MD', 'Baltimore MD', 'SHIP']
    # labels_temperature = ["FSKM2", "Baltimore MD", "Baltimore-Martin MD", "Gaithersburg", "TCBM2", "Patapsco, MD", "BUOY", "Carroll County"]
    labels_temperature = ['Carroll County', 'FSKM2', 'Gaithersburg', 'TCBM2', 'Baltimore-Martin MD', 'Baltimore MD', 'FSNM2', 'BLTM2'] 
    csv_f = "D:/project/Data_tree.csv"
    json_labels = ["temperature", "dewpoint_spread", "dewpoint", "pressure", "wetbulbtemp"]
    locations = {}

    data_dict = {
        "pressure": labels_pressure,
        "dewpoint": labels_dewpoint,
        "dps": labels_dewpoint_spread,
        "wbt": labels_wetbulbtemp,
        "temp":labels_temperature
    }
    
    label = "temperature"
    print(label)
    

    with open("/data_new_tree.json", "r") as f: 
        data = json.load(f)


    # Used to list data from the json file 
    """
    
    for t in data: # t = time and date
        for i in data[t]:
            # if i != "precipitation" and i != "picture":
            if i == label:
                if data[t][i] != None: 
                    for element in data[t][i]:
                        # print(element)
                        amount, location = parseTree(element)
                        if location not in locations:
                            locations[location] = 1
                        else: 
                            locations[location] += 1
                        
                            
    """

    # 30 elements
    header = ['FSNM2_pressure', 'Baltimore MD_pressure', 'Gaithersburg_pressure', 'SHIP_pressure', 'FSKM2_pressure', 
              'BLTM2_pressure', 'TCBM2_pressure', 'Baltimore-Martin MD_pressure', 'Carroll County_dewpoint', 'Baltimore-Martin MD_dewpoint', 
              'Gaithersburg_dewpoint', 'Baltimore MD_dewpoint', 'SHIP_dewpoint', 'Gaithersburg_dps', 'Baltimore MD_dps',
              'Carroll County_dps', 'Baltimore-Martin MD_dps', 'SHIP_dps', 'Gaithersburg_wbt', 'Baltimore-Martin MD_wbt', 
              'Baltimore MD_wbt', 'SHIP_wbt', 'Carroll County_temp', 'FSKM2_temp', 'Gaithersburg_temp', 
              'TCBM2_temp', 'Baltimore-Martin MD_temp', 'Baltimore MD_temp', 'FSNM2_temp', 'BLTM2_temp', "time"]
    
    print(f"Header len: {len(header)}")
    with open(csv_f, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # for x in data_dict:
        #     for l in data_dict[x]:
                # header.append(l + "_" + x)
                # print(l)
        
        writer.writerow(header)
        total_valid_data = 0
        total_data = 0
        for t in data: 
            row = 31 * [None]
            for l in data[t]:
                if l in json_labels:
                    for item in data[t][l]:
                        amount, location = parseTree(item)
                        amount = amount.replace(',','')
                        # print(f"Amount: {amount}")
                        # print(f"Location: {location}")

                        #json_labels = ["temperature", "dewpoint_spread", "dewpoint", "pressure", "wetbulbtemp"]

                        if l == json_labels[0]: 
                            if location in data_dict["temp"]:
                                hl = location + "_" + "temp"
                                indx = header.index(hl)                                
                                row[indx] = amount
                        elif l == json_labels[1]:
                            if location in data_dict["dps"]:
                                hl = location + "_" + "dps"
                                indx = header.index(hl)
                                row[indx] = amount
                        elif l == json_labels[2]:
                            if location in data_dict["dewpoint"]:
                                hl = location + "_" + "dewpoint"
                                indx = header.index(hl)
                                row[indx] = amount
                        elif l == json_labels[3]:
                            if location in data_dict["pressure"]:
                                hl = location + "_" + "pressure"
                                indx = header.index(hl)
                                row[indx] = amount
                        elif l == json_labels[4]:
                            if location in data_dict["wbt"]:
                                hl = location + "_" + "wbt" 
                                indx = header.index(hl)
                                row[indx] = amount

                        else:
                            print(f"Invalid Label: {l}")
            # print(f"Time: {t}")
            row_c = 0
            for n in row:
                if n == None:
                    row_c += 1
            if row_c >= 9: # total valid data = 9181 total data: 62189
                pass
            else:
                # print(row) 
                row[30] = t
                writer.writerow(row)
                total_valid_data +=1
            total_data += 1
        print(f"Total Data: {total_data}")
        print(f"Total valid Data: {total_valid_data}")
            
            
# K-Nearest Neighbor (KNN) with N = 5 (fill in missing data)
# Data_tree_imputed is only the tree data with the imputed data added in 
# data_tree_full is all the data_tree_imputed data + precipitation data 
def imputate():
    data = pd.read_csv('./Data_tree.csv')
    # data = data.iloc[:, :-1]
    dates = data.pop("time")
    missing_columns = data.columns[data.isna().any()]
    imputer = KNNImputer(n_neighbors=5)
    data_imputed = imputer.fit_transform(data)
    data_imputed_df = pd.DataFrame(data_imputed, columns=data.columns)
    data = data_imputed_df
    data.insert(len(data.columns), "time", dates)
    data.to_csv('./Data_tree_imputed.csv', index=False)
    print("Done")

        
# Add precipitation data with the imputated data (all data is now in 1 file)  
def Addprecipitation():
    label_r = ["Carroll County", "Baltimore MD", "Gaithersburg", "Baltimore-Martin MD","NoRain", "Image"]
    with open("./data_new.json", "r") as f: 
        data = json.load(f)
    file_path = "./Data_tree_imputed.csv"
    count = 0
    df = pd.read_csv(file_path)
    precipitation = len(df) * [None]
    print(len(df))
    date_column = df.iloc[:, -1]
    for d in date_column:
        if d in data: 
            row = [0, 0, 0, 0, 0]
            if "precipitation" in data[d]:
                row_index = df.index[df['time'] ==d].tolist()[0]
                col_index = df.columns.get_loc("time")
                count += 1
                for place in data[d]["precipitation"]:
                    amount, location = parsetext(place)
                    # print(amount, location)
                    location = location.strip()
                    # print(f"Location: {location} Amount: {amount}")

                    if amount > 0: 
                        amount = int(1)
                    else: 
                        amount = int(0)
                    if location == label_r[0]:
                        row[0] = amount
                    elif location == label_r[1]:
                        row[1] = amount
                    elif location == label_r[2]:
                        row[2] = amount
                    elif location == label_r[3]: 
                        row[3] = amount
                    else:
                        print(f"Error unknown location: {location}")
                # print(write_list)
                if row[0] == 0 and row[1] == 0 and row[2] == 0 and row[3] == 0:
                    row[4] = 1
                
                precipitation[row_index] = row
                # print(row)

    df.insert(len(df.columns), "precipitation", precipitation)
    df_cleaned = df.dropna()
    print(df_cleaned)
    df_cleaned.to_csv('./Data_tree_full.csv', index=False)

      
             
    print(f"Count: {count}")

    
    

    
    # Used to get the labels to use for the data
    """
    
    total = 0
    mean = 0
    for x in locations: 
        total += locations[x]
        
    
    mean = (total / len(locations)) 
    valid = []
    print(f"Mean: {mean}")
    for x in locations: 
        if locations[x] >= mean: 
            print(f"{x}: {locations[x]}")
            valid.append(x)
    print(valid)
            

    
    
    """
    


            
            


    






if __name__ == '__main__':
    # cleanup()
    # cleanupTree()
    # imputate()
    # Addprecipitation()
    pass
    