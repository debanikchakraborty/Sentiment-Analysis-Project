# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 22:11:19 2023

@author: deban
"""
#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#load the data into dataframe
df = pd.read_csv("C:\\Users\\deban\\OneDrive - Oklahoma A and M System\\Fall 2023\\Programming for Data Science 1\\Final Report\\Dataset\\SampleAppliance_Output.csv")
print(df)

#Rename normalized google api score to google api score for better understanding
df.rename(columns={'Normalized_Sentiment_Scores': 'Google_API_Score'}, inplace=True)

#importing ggplot for better plotting
plt.style.use('ggplot')

#importing and downloading necessary NLTK librarbies and functions
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

example= df['reviewText'][32] #Setting an example to conduct testing


#VADER (Bag of Words Approach)- scores sentiment of each word and then combine them to score them neg, pos and neutral, 
#stop words(and,the) are removed, doesnt account for relationship between words
#importing some necessary libraries for VADER coding
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm #progress Bar Tracker for looping

#Creating the funciton to feed text to derive VADER scores of it
nltk.download('vader_lexicon')
sia= SentimentIntensityAnalyzer()

#Running some examples to check their scores
sia.polarity_scores(example)

#RoBERTa Model-Considers the relationship in between words, 
#trained with large corpus of data, accounts for the words but also context related to other words
#Use Hugging Face
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax #applied to the outputs to scale down to 0 and 1

#Importing a model trained on bunch of twitter comments that were labled
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
#Pulled down the trained weights for that twitter project, transfer learning 
model = AutoModelForSequenceClassification.from_pretrained(MODEL) 

#Applying the model tokenizer to an example and detaching the scores from an array
encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
}
print(scores_dict)

#Creating the function to assign the roberta score to a text
def polarity_scores_roberta(example1):
    encoded_text = tokenizer(example1, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict

#Making an empty dictionary to save the Vader and Roberta scores
result={}
for i, row in tqdm(df.iterrows(), total=len(df)): #looping through each row in df upto its total length
    #RoBERTa as a trained model has a word limit to score a text,
    #so it faced runtime error with some of the longest reviews. 
    #we used try and except in the loop to ignore those longest reviews (which were very few)
    try: 
        text = row['reviewText'] #taking each review as the text
        myid = row['Serial No'] #taking associated serial no of the row as an id
        vader_result= sia.polarity_scores(text) #Applying VADER polarity score function to each text
        vader_result_rename = {} #rename vader score columns
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(text) #Applying created ROBERTA score function on each text
        both = {**vader_result_rename, **roberta_result} #attaching VADER & ROBERTA scores in a dictionary
        result[myid] = both #arraning the result from previous dictionary based on id of each row in result dictionary
    except RuntimeError:
        #print the Serial No of the lognest reviews which had runtime errors (only 9 out of 1993)
        print(f'Broke for Serial No{myid}') 

#Making result dataframe from the dictionary and transposing it
result_df=pd.DataFrame(result).T
result_df=result_df.reset_index().rename(columns={'index':'Serial No'}) #Serial No as a column to use it as primary key for the join
#left joining with main dataframe to create final_df using Serial No as primary key
final_df= df.merge(result_df, how='left')
#dropping the rows for which we could not obtain vader and roberta scores 
final_df = final_df.dropna(subset=['vader_neg','vader_neu', 'vader_neg','vader_compound','roberta_pos','roberta_neu','roberta_neg'])
#calcualte the roberta compound score based on a formula with its pos, neg and neu scores
final_df['roberta_compound'] = final_df['roberta_neg'] + (final_df['roberta_neu']*2) + (final_df['roberta_pos']*3) - 2
print(final_df)
#Saving the output to a csv file
final_df.to_csv('C:\\Users\\deban\\OneDrive - Oklahoma A and M System\\Fall 2023\\Programming for Data Science 1\\Final Report\\Dataset\\Finalsample_file.csv', index=False)

