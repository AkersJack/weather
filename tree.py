from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import pandas as pd 
import os
import ast
import random
import time
import matplotlib.pyplot as plt

def newTest():
    location = "./Data_tree_full_4.csv"
    label_r = ["Carroll County", "Baltimore MD", "Gaithersburg", "Baltimore-Martin MD","NoRain"]
    cnn_vals = []
    file = open("Predict", "r")
    lines = file.readlines()
    # print(f"Lines len: {len(lines)}")
    for x in lines: 
        line = x.strip('\n') 
        line = ast.literal_eval(line)
        cnn_vals.append(line)
        # print(line)
    # print(len(cnn_vals))

    df = pd.read_csv(location)
    y = df.iloc[:,-1]
    X = df.iloc[:, 0:30]
    # print(df["precipitation"])
    new_val = [[], [], [], []]
    
    for li, x in enumerate(df["precipitation"]):
        x_list = ast.literal_eval(x)
        for i, v in enumerate(x_list):
            val = cnn_vals[li][i]
            
            new_val[i].append(val)
        

    
    for i, v in enumerate(new_val):
        X.insert(len(X.columns), column=label_r[i], value=v)

    for i, item in enumerate(y): 
        y.at[i] = ast.literal_eval(item)
    
    Y = {
        label_r[0]:[],
        label_r[1]:[],
        label_r[2]:[],
        label_r[3]:[],
    }
    keys = Y.keys()
    keys = list(keys)
    for x in y: 

        for i, v in enumerate(x): 
            Y[keys[i]].append(v)
     
    Y = pd.DataFrame(Y)

    
        

        


    
    return X, Y

def loaddata():
    location = "./Data_tree_full_4.csv"
    label_r = ["Carroll County", "Baltimore MD", "Gaithersburg", "Baltimore-Martin MD","NoRain"]
    cnn_vals = []
    file = open("Predict", "r")
    lines = file.readlines()
    # print(f"Lines len: {len(lines)}")
    for x in lines: 
        line = x.strip('\n') 
        line = ast.literal_eval(line)
        cnn_vals.append(line)

    df = pd.read_csv(location)
    y = df.iloc[:,-1]
    X = df.iloc[:, 0:30]
    # print(df["precipitation"])
    new_val = [[], [], [], []]
    
    for li, x in enumerate(df["precipitation"]):
        x_list = ast.literal_eval(x)
        for i, v in enumerate(x_list):
            val = v
            # print(f"Value: {val}")
            
            new_val[i].append(val)
        

    
    for i, v in enumerate(new_val):
        X.insert(len(X.columns), column=label_r[i], value=v)

    for i, item in enumerate(y): 
        y.at[i] = ast.literal_eval(item)
    
    Y = {
        label_r[0]:[],
        label_r[1]:[],
        label_r[2]:[],
        label_r[3]:[],
    }
    keys = Y.keys()
    keys = list(keys)
    for x in y: 

        for i, v in enumerate(x): 
            Y[keys[i]].append(v)
     
    Y = pd.DataFrame(Y)

    # for r, item in Y.iterrows(): 
        # print(item)
    
        

        


    
    return X, Y
    
             
                
















if __name__ == '__main__':
    X, y = loaddata()
    test_x, test_y = newTest()
    
 
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    

    # Step 4: Create a Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the classifier
    # rf_classifier.fit(X_train, y_train)
    rf_classifier.fit(X, y)

    feature_importances = rf_classifier.feature_importances_
    print(f"Feature importances: {feature_importances}")


    # Make predictions on the test set
    # y_pred = rf_classifier.predict(X_test)
    y_pred = rf_classifier.predict(test_x)

    accuracy = accuracy_score(test_y, y_pred)
    print('\n')
    print(f"Accuracy: {accuracy:.2f}")
    print('\n')
    
    # Create a bar chart to visualize feature importance
    plt.figure(figsize=(8, 6))
    plt.barh(X.columns, feature_importances, color='skyblue')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Name')
    plt.title('Random Forest Feature Importance')
    plt.gca().invert_yaxis()  # This will display the petal features at the top
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()



