# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 02:35:34 2023

@author: deban
"""
#importing necessary libraries
import pandas as pd
import numpy as np


#load the main data into dataframe (10,610 rows)
df = pd.read_csv("C:\\Users\\deban\\OneDrive - Oklahoma A and M System\\Fall 2023\\Programming for Data Science 1\\Final Report\\Dataset\Excel Files\\Concise_ApplianceDataset.csv")
print(df)
#importing and downloading necessary NLTK librarbies and functions
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

#RoBERTa Model-Considers the relationship in between words, 
#trained with large corpus of data, accounts for the words but also context related to other words
#Use Hugging Face
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax #applied to the outputs to scale down to 0 and 1
from tqdm.notebook import tqdm #progress Bar Tracker for looping

#Importing a model trained on bunch of twitter comments that were labled
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
#Pulled down the trained weights for that twitter project, transfer learning 
model = AutoModelForSequenceClassification.from_pretrained(MODEL) 

#Creating the function to assign the roberta score to a text
def polarity_scores_roberta(text):
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict

#Making an empty dictionary to save the Roberta scores
result={}
for i, row in tqdm(df.iterrows(), total=len(df)): #looping through each row in subdf1 upto its total length
    #RoBERTa as a trained model has a word limit to score a text,
    #so it faced runtime error with some of the longest reviews. 
    #we used try and except in the loop to ignore those longest reviews (which were very few)
    try: 
        text = row['reviewText'] #taking each review as the text
        myid = row['Serial No'] #taking associated serial no of the row as an id
        roberta_result = polarity_scores_roberta(text) #Applying created ROBERTA score function on each text
        result[myid] = roberta_result #arraning the result of roberta scoring based on id of each row in result dictionary
    except RuntimeError:
        #print the Serial No of the lognest reviews which had runtime errors (only 41 out of 5305)
        print(f'Broke for Serial No{myid}') 

#faced an index error after 8515 rows, so starting from 8516th row
sub=df[8516:] #create a subdf with all rows after 8516th row (total 2094 rows)

naal={} #save roberta score for rest of the rows in a new dictionary
for i, row in tqdm(sub.iterrows(), total=len(sub)): #looping through each row in sub df upto its total length
    #RoBERTa as a trained model has a word limit to score a text,
    #so it faced runtime error with some of the longest reviews. 
    #we used try and except in the loop to ignore those longest reviews (which were very few)
    try: 
        text = row['reviewText'] #taking each review as the text
        myid = row['Serial No'] #taking associated serial no of the row as an id
        roberta_result = polarity_scores_roberta(text) #Applying created ROBERTA score function on each text
        naal[myid] = roberta_result #arraning the result of roberta scoring based on id of each row in result dictionary
    except RuntimeError:
        #print the Serial No of the lognest reviews which had runtime errors (only 4 out of 2094)
        print(f'Broke for Serial No{myid}')         

#Making result dataframe from the dictionary and transposing it
results= {**result, **naal} #merging two dictionary

result_df=pd.DataFrame(results).T #creating a dataframe with the scores
len(result_df) 
#We are getting RoBERTa scores for total 10,565 rows as we faced total runtime error for 45 rows

result_df=result_df.reset_index().rename(columns={'index':'Serial No'}) #Serial No as a column to use it as primary key for the join
#left joining with main dataframe to create final_df using Serial No as primary key
final_df= df.merge(result_df, how='left')
final_df.isnull().sum() #checking if we have 45 null values for roberta scores to confirm

#dropping the rows for which we could not obtain vader and roberta scores 
final_df = final_df.dropna(subset=['roberta_pos','roberta_neu','roberta_neg'])
final_df.isnull().sum() #checking again if we have any more null values

#calcualte the roberta compound score based on a formula with its pos, neg and neu scores
final_df['roberta_compound'] = final_df['roberta_neg'] + (final_df['roberta_neu']*2) + (final_df['roberta_pos']*3) - 2
print(final_df)
#Saving the output to a csv file
final_df.to_csv('C:\\Users\\deban\\OneDrive - Oklahoma A and M System\\Fall 2023\\Programming for Data Science 1\\Final Report\\Dataset\\Coded_whole_dataset.csv', index=False)

