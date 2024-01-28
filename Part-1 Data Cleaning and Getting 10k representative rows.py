# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:19:27 2023

@author: deban
"""

import pandas as pd
import numpy as np
# Load the CSV file
df = pd.read_excel("C:\\Users\\deban\\OneDrive - Oklahoma A and M System\\Fall 2023\\Programming for Data Science 1\\Final Report\\Dataset\\ApplianceDataset.xlsx")

#dropping the duplicate rows for review text
bf = df.drop_duplicates(subset=['reviewText']) 
len(df)-len(bf) #15300 duplicate rows dropped

#Keeping the Verified=True reviews only
gf=bf[bf['verified']==True]  
len(bf)-len(gf)#4699 unverified reviews dropped

df=gf #set df as the updated dataframe
len(df) # we have 70739 in the updated dataframe

#Percentages of each ratings in the dataset
dfcatrat=df.groupby('ratings').agg(Counts=('ratings', 'size'))
dfcatrat['total']=dfcatrat['Counts'].sum()
dfcatrat['percent']=round((dfcatrat['Counts']/dfcatrat['total']),4)
print(dfcatrat)

#Percentages of each product types in the dataset (each stratum)
dfcatprod=df.groupby('product type').agg(Counts=('product type', 'size')).reset_index()
dfcatprod['total']=dfcatprod['Counts'].sum()
dfcatprod['percent']=round((dfcatprod['Counts']/dfcatprod['total']),4)
print(dfcatprod)

#Percentages of each product type's each rating in the dataset
dfcat=df.groupby(['product type','ratings'])[['product type','ratings']].agg(count=('ratings','size')).reset_index()
dfcat['Percentage']=dfcat['count']/len(df)
print(dfcat)

#Fixing how much percentage of rows we want to drop, 
#our target is to get around 11000 rows, so we drop 85% of 70739 rows
percentage_to_drop = 0.85

product = dfcatprod["product type"].tolist() #Create a list of 42 product types that we have


for cat in product: #making a loop on all the product types
    # Create a mask to identify rows with the each specific product type and rating 1 (one stratum)
    mask = (df["product type"] == cat) & (df['ratings'] == 1) 
    # Get the indices of the rows which are randomly dropped from that particular cluster
    indices_to_drop = df[mask].sample(frac=percentage_to_drop).index
    # Drop the selected rows from the DataFrame
    df.drop(indices_to_drop, inplace=True)
    # Create a mask to identify rows with the each specific product type and rating 2 (another stratum)
    mask = (df["product type"] == cat) & (df['ratings'] == 2)
    # Get the indices of the rows which are randomly dropped from that particular cluster
    indices_to_drop = df[mask].sample(frac=percentage_to_drop).index
    df.drop(indices_to_drop, inplace=True)
    #Same mechanism for cluster of each product type with rating 3
    mask = (df["product type"] == cat) & (df['ratings'] == 3)
    indices_to_drop = df[mask].sample(frac=percentage_to_drop).index
    df.drop(indices_to_drop, inplace=True)
    #Same mechanism for cluster of each product type with rating 4
    mask = (df["product type"] == cat) & (df['ratings'] == 4)
    indices_to_drop = df[mask].sample(frac=percentage_to_drop).index
    df.drop(indices_to_drop, inplace=True)
    #Same mechanism for cluster of each product type with rating 5
    mask = (df["product type"] == cat) & (df['ratings'] == 5)
    indices_to_drop = df[mask].sample(frac=percentage_to_drop).index
    df.drop(indices_to_drop, inplace=True)

len(df) #we have 10,610 rows now

#Let's see if they are the proper representative of of the percentages of each stratum (each product's each rating)

#Check the Percentages of each product types in the dataset and compare with previous
dfcatprod2=df.groupby('product type').agg(Counts=('product type', 'size')).reset_index()
dfcatprod2['total']=dfcatprod2['Counts'].sum()
dfcatprod2['percent']=round((dfcatprod2['Counts']/dfcatprod2['total']),4)
print(dfcatprod2)

#Check the Percentages of each ratings in the dataset and compare with previous
dfcatrat2=df.groupby('ratings').agg(Counts=('ratings', 'size'))
dfcatrat2['total']=dfcatrat2['Counts'].sum()
dfcatrat2['percent']=round((dfcatrat2['Counts']/dfcatrat2['total']),4)

# Check the Percentages of each product type's each rating in the dataset and compare it with previous
dfcat2=df.groupby(['product type','ratings'])[['product type','ratings']].agg(count=('ratings','size')).reset_index()
dfcat2['Percentage']=dfcat2['count']/len(df)

#Comparing the percentages of the each product type's each rating between the two datasets, 
#we find that the new dataset is not very different from the previous dataset. 
#It is a proper representation of the previous dataset

#adjusting the serial number of the rows for the new dataset
df=df.drop('serial No', axis=1)
bf=df.reset_index()
bf=bf.drop('index', axis=1)
bf['Serial No'] = bf.index
column_to_move='Serial No'
#bringing the serial no to left most of the dataset
bf = pd.concat([bf[column_to_move], bf.drop(column_to_move, axis=1)], axis=1)

# Specify the file name and set index to False to exclude the index column
bf.to_csv('Concise_ApplianceDataset.csv', index=False)  


