# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:59:54 2023

@author: deban
"""
#importing json, gzip and other necessary libraries
import json
import gzip
import pandas as pd
from urllib.request import urlopen


### load the meta data (meta data contains product catagory and type related information)
data = []
with gzip.open("C:\\Users\\deban\\Downloads\\meta_Appliances.json.gz") as f:
    for l in f:
        data.append(json.loads(l.strip()))
    
# total length of list, this number equals total number of products
print(len(data))

# first row of the list
print(data[0])

# convert list into pandas dataframe

meta = pd.DataFrame.from_dict(data)

#keeping only catagory, brand, main cat, price and asin of eac
meta= meta[['category','brand','main_cat', 'price', 'asin']]


#load the Aoppliance Dataset (this data set contains the actual reviews of those product types)
#reviewText, reviewer name, summary of the review, review time
data = []
with gzip.open("C:\\Users\\deban\\Downloads\\Appliances.json.gz") as f:
    for l in f:
        data.append(json.loads(l.strip()))
    
# total length of list, this number equals total number of products
print(len(data))

# first row of the list
print(data[0])

# convert list into pandas dataframe

review = pd.DataFrame.from_dict(data)

print(len(review))

#left joining meta dataset to appliance dataset(which has the reviews) using asin as the key
#asin is the unqiue product code
df_left = pd.merge(review, meta, left_on='asin',right_on='asin', how='inner')
pd.set_option('display.max_columns', None) #show all columns
print(df_left)

#drop the null values
final=df_left.dropna()

#dropping the rows with null values for specific rows
final = df_left.dropna(subset=['category','brand','main_cat','price'])

#converting the final dataframe to csv file
final.to_csv('appliancefinal.csv')
