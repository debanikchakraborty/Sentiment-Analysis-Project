# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 01:32:11 2023

@author: deban
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

final_df= pd.read_csv("C:\\Users\\deban\\OneDrive - Oklahoma A and M System\\Fall 2023\\Programming for Data Science 1\\Final Report\\Dataset\\Excel Files\\Finalsample_file.csv")

#Evaluation Criteria One: Comparison with GOLD STANDARD (Human Sentiment Score)
#Vizualize the True Score Distribution
plt.figure(figsize=(6, 4))
plt.hist(final_df['Human Sentiment Score'], bins='auto', edgecolor='black')  # 'auto' determines the number of bins automatically
plt.title('Histogram of TRUE Sentiment Score')
plt.xlabel('True Sentiment score')
plt.ylabel('Frequency')

#Vizualize the VADER Score Distribution
plt.figure(figsize=(6, 4))
plt.hist(final_df['vader_compound'], bins='auto', edgecolor='black')  # 'auto' determines the number of bins automatically
plt.title('Histogram of VADER Sentiment Score')
plt.xlabel('VADER Compound Sentiment score')
plt.ylabel('Frequency')

#Vizualize the Roberta Score Distribution
plt.figure(figsize=(6, 4))
plt.hist(final_df['roberta_compound'], bins='auto', edgecolor='black')  # 'auto' determines the number of bins automatically
plt.title('Histogram of RoBERTa Sentiment Score')
plt.xlabel('RoBERTa Compound Sentiment score')
plt.ylabel('Frequency')

#Vizualize the Google API Score Distribution
plt.figure(figsize=(6, 4))
plt.hist(final_df['Google_API_Score'], bins='auto', edgecolor='black')  # 'auto' determines the number of bins automatically
plt.title('Histogram of Google API Sentiment Score')
plt.xlabel('Google API Sentiment score')
plt.ylabel('Frequency')

#Scatter plot between VADER Comp, ROBERTA Comp and Google API scores and Human Sentiment Scores
# Create subplots with 1 row and 3 columns
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Scatter plot for VADER Compound Sentiment Score
axes[0].scatter(final_df['Human Sentiment Score'], final_df['vader_compound'], color='blue', marker='o', label='VADER Compound')
axes[0].set_title('VADER Compound')
axes[0].set_xlabel('Human Sentiment Score')
axes[0].set_ylabel('VADER Compound Score')
axes[0].legend()

# Scatter plot for RoBERTa Compound Sentiment Score
axes[1].scatter(final_df['Human Sentiment Score'], final_df['roberta_compound'], color='red', marker='o', label='RoBERTa Compound')
axes[1].set_title('RoBERTa Compound')
axes[1].set_xlabel('Human Sentiment Score')
axes[1].set_ylabel('RoBERTa Compound Score')
axes[1].legend()

# Scatter plot for Google API Sentiment Score
axes[2].scatter(final_df['Human Sentiment Score'], final_df['Google_API_Score'], color='green', marker='o', label='Google API')
axes[2].set_title('Google API')
axes[2].set_xlabel('Human Sentiment Score')
axes[2].set_ylabel('Google API Score')
axes[2].legend()

# Adjust layout for better spacing
plt.tight_layout()


#Correlation with Human Sentiment Scores
#VADER correlation to True score= 0.55
v=final_df['vader_compound'].corr(final_df['Human Sentiment Score'])

#ROBERTA correlation to True score= 0.82
r=final_df['roberta_compound'].corr(final_df['Human Sentiment Score'])

#Google API correlation to True score= 0.69
g=final_df['Google_API_Score'].corr(final_df['Human Sentiment Score'])

# Sample data
data = [['VADER', round(v,2)],
        ['RoBERTa', round(r,2)],
        ['Google API', round(g,2)]]

# Define column headers
headers = ['Model', 'Correlation']

# Display the table
title = 'Correlation to True Sentiment Scores'
print(title)
table = tabulate(data, headers=headers, tablefmt='grid')

print(table)
print('\n')
#-----------------------------------------------------------------------------------------

#Evaluation Criteria Two: Relationship with Ratings
#Distribution of Ratings

#Human Sentiment Score vs Ratings
plt.figure(figsize=(8, 6))
sns.barplot(data=final_df, x='ratings', y='Human Sentiment Score')
plt.title('True Sentiment score by Review Ratings')
plt.show()

#VADER Score relationship with ratings 
plt.figure(figsize=(8, 6))
sns.barplot(data=final_df, x='ratings', y='vader_compound')
plt.title('VADER Compound score by Review Ratings')
plt.show()

#ROBERTA Score relationship with ratings 
plt.figure(figsize=(8, 6))
sns.barplot(data=final_df, x='ratings', y='roberta_compound')
plt.title('RoBERTa Compound score by Review Ratings')
plt.show()
    
#Google API Score relationship with ratings
plt.figure(figsize=(8, 6)) 
sns.barplot(data=final_df, x='ratings', y='Google_API_Score')
plt.title('Google API score by Review Ratings')
plt.show()


#Correlation with Ratings
#VADER score correlation to Ratings= 0.50
v=final_df['vader_compound'].corr(final_df['ratings'])

#RoBERTa score correlation to Ratings= 0.78
r=final_df['roberta_compound'].corr(final_df['ratings'])

#Google API score correlation to Ratings= 0.67
g=final_df['Google_API_Score'].corr(final_df['ratings'])

# Sample data
data = [['VADER', round(v,2)],
        ['RoBERTa', round(r,2)],
        ['Google API', round(g,2)]]

# Define column headers
headers = ['Model', 'Correlation']

# Display the table
title = 'Correlation to Ratings'
print(title)
table = tabulate(data, headers=headers, tablefmt='grid')


print(table)
print('\n')
#-----------------------------------------------------------------------------


#Evaluation Criteria Three: Precision, Recall and F1 scores

#VADER Evaluation:
#True Positive, False Positive, True Negative, False Negative Calculation
final_df[(final_df['vader_compound']<0.25) & (final_df['vader_compound']>-0.25)] #checking neutral rows 376 rows
tp1=len(final_df[(final_df['Review Type'] == 'Positive') & (final_df['vader_compound']>0.25)])
fp1=len(final_df[(final_df['Review Type'] == 'Negative') & (final_df['vader_compound']>0.25)])
tn1=len(final_df[(final_df['Review Type'] == 'Negative') & (final_df['vader_compound']<-0.25)])
fn1=len(final_df[(final_df['Review Type'] == 'Positive') & (final_df['vader_compound']<-0.25)])
#Precision, Recall and F1 Score Calculation
Precision1= tp1/(tp1+fp1) #0.93
Recall1= tp1/(tp1+fn1) #0.95
F11= (2*Precision1*Recall1)/(Precision1+Recall1) #0.94


#ROBERTA Evaluation:
#True Positive, False Positive, True Negative, False Negative Calculation
final_df[(final_df['roberta_compound']<0.25) & (final_df['roberta_compound']>-0.25)] #210 neutral rows
tp2=len(final_df[(final_df['Review Type'] == 'Positive') & (final_df['roberta_compound']>0.25)])
fp2=len(final_df[(final_df['Review Type'] == 'Negative') & (final_df['roberta_compound']>0.25)])
tn2=len(final_df[(final_df['Review Type'] == 'Negative') & (final_df['roberta_compound']<-0.25)])
fn2=len(final_df[(final_df['Review Type'] == 'Positive') & (final_df['roberta_compound']<-0.25)])
#Precision, Recall and F1 Score Calculation
Precision2= tp2/(tp2+fp2) #0.99
Recall2= tp2/(tp2+fn2) #0.96
F12= (2*Precision2*Recall2)/(Precision2+Recall2) #0.97

#Google API Evaluation
final_df[(final_df['Google_API_Score']<0.25) & (final_df['Google_API_Score']>-0.25)] #619 neutral rows
#True Positive, False Positive, True Negative, False Negative Calculation
tp3=len(final_df[(final_df['Review Type'] == 'Positive') & (final_df['Google_API_Score']>0.25)])
fp3=len(final_df[(final_df['Review Type'] == 'Negative') & (final_df['Google_API_Score']>0.25)])
tn3=len(final_df[(final_df['Review Type'] == 'Negative') & (final_df['Google_API_Score']<-0.25)])
fn3=len(final_df[(final_df['Review Type'] == 'Positive') & (final_df['Google_API_Score']<-0.25)])
#Precision, Recall and F1 Score Calculation
Precision3= tp3/(tp3+fp3) #0.99
Recall3= tp3/(tp3+fn3) #0.97
F13= 2*((Precision3*Recall3)/(Precision3+Recall3)) #0.98

#Make a table for all models' True Positive, True Negative, False Positive and False Negative
# Sample data
data = [['VADER', tp1, fp1, tn1, fn1],
        ['RoBERTa', tp2, fp2, tn2, fn2],
        ['Google API', tp3, fp3, tn3, fn3]]

# Define column headers
headers = ['Model', 'True Positive','False Positive', 'True Negative', 'False Negative']

# Display the table
title = 'Counts of True and False Positive & True and False Negative'
print(title)

table = tabulate(data, headers=headers, tablefmt='grid')

print(table)
print('\n')


#Make a table for all models' Precision, Recall and F-1 Score
# Sample data
data = [['VADER', round(Precision1,2), round(Recall1,2), round(F11,2)],
        ['RoBERTa', round(Precision2,2), round(Recall2,2), round(F12,2)],
        ['Google API', round(Precision3,2), round(Recall3,2), round(F13,2)]]

# Define column headers
headers = ['Model', 'Precision', 'Recall','F1-Score']

# Display the table
title = 'Precision, Recall and F1 Score'
print(title)

table = tabulate(data, headers=headers, tablefmt='grid')

print(table)