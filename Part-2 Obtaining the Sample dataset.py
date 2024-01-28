# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 16:33:15 2023

@author: deban
"""

import pandas as pd
import numpy as np

df = pd.read_csv("C:\\Users\\deban\\OneDrive - Oklahoma A and M System\\Fall 2023\\Programming for Data Science 1\\Final Report\\Dataset\\Concise_ApplianceDataset.csv")

#dropping some null values for other columns, except vote as its number of null values are pretty huge
df.isnull().sum()
df=df.dropna(subset=['reviewerName'])
df=df.dropna(subset=['summary'])
df=df.dropna(subset=['brand'])
df.isnull().sum()

#Each Stratum = Each Product Type's Each Rating
#Percentages of the strata in the dataset
dfcat=df.groupby(['product type','ratings'])[['product type','ratings']].agg(count=('ratings','size')).reset_index()
dfcat['Percentage']=dfcat['count']/len(df)
print(dfcat)

bf=df.groupby('product type').agg(Counts=('product type', 'size')).reset_index() #making a list of the product types
product = bf["product type"].tolist() #creating a list of the 42 products

# we want a sample of 2000 rows, so we are randomly dropping around 80% from each stratum of the dataset
# so the concept is overall 80% rows will be dropped from the entire dataset
#but the percentage of each stratum will be similar as before indicating a proper stratified sampling
percentage_to_drop = 0.812

for cat in product: #making a loop on all the product types
    # Create a mask to identify rows with the each specific product type and rating 1 (one cluster)
    mask = (df["product type"] == cat) & (df['ratings'] == 1) 
    # Get the indices of the rows which are randomly dropped from that particular cluster
    indices_to_drop = df[mask].sample(frac=percentage_to_drop).index
    # Drop the selected rows from the DataFrame
    df.drop(indices_to_drop, inplace=True)
    # Create a mask to identify rows with the each specific product type and rating 2 (one cluster)
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

len(df) #we have a sample of 1993 rows 

# Check the Percentages of each product type's each rating (each stratum) in sample and compare it with previous dataset
dfcat2=df.groupby(['product type','ratings'])[['product type','ratings']].agg(count=('ratings','size')).reset_index()
dfcat2['Percentage']=dfcat2['count']/len(df)
print(dfcat)

#the percentages from each stratum are almost same, so the new dataframe is a proper stratified sample

#Specify the file name and set index to False to exclude the index column
df.to_csv('sample_appliance.csv',index=False) 