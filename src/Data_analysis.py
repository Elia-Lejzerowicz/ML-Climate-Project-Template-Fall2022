import pandas as pd
import os
import csv

#f = open("/Users/elialejzerowicz/climate/ML-Climate-Project-Template-Fall2022/Data_website.csv")

#data_website = pd.read_csv("Data_website.csv")

with open('/Users/elialejzerowicz/climate/ML-Climate-Project-Template-Fall2022/Data_website.csv','r') as file:
    employee = csv.reader(file)

#print(data_website.head(10))