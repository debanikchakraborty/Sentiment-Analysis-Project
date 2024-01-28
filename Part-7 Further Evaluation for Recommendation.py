# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 00:09:16 2023

@author: deban
"""
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

final_df = pd.read_csv('C:\\Users\\deban\\OneDrive - Oklahoma A and M System\\Fall 2023\\Programming for Data Science 1\\Final Report\\Dataset\\Excel Files\\Coded_whole_dataset.csv')
print(final_df)
#converting price column to numeric as needed for further analysis
final_df['price'] = pd.to_numeric(final_df['price'], errors='coerce')
#---------------------------------------------------------------------------------
#Checking Some Attributes of our sentiment scores

#Vizualize the RoBEERTa Score Distribution
plt.figure(figsize=(6, 4))
plt.hist(final_df['roberta_compound'], bins='auto', edgecolor='black')  # 'auto' determines the number of bins automatically
plt.title('Histogram of RoBERTa Sentiment Score')
plt.xlabel('RoBERTa Sentiment score')
plt.ylabel('Frequency')

#Vizualize the RoBEERTa Score relationship with Ratings
plt.figure(figsize=(8, 6))
sns.barplot(data=final_df, x='ratings', y='roberta_compound')
plt.title('RoBERTa Sentiment Score by Ratings')
plt.show()

#Correlation of Score to Ratings
correlation= final_df['roberta_compound'].corr(final_df['ratings']) 
print('Correlation of RoBERTa scoring of our dataset with ratings is', round(correlation,2)) #0.76

#Vizualize the word clouds for positive and negative reviews
# Separate positive and negative reviews based on the roberta sentiment score
positive_reviews = final_df[final_df['roberta_compound'] > 0]['reviewText'].str.cat(sep=' ')
negative_reviews = final_df[final_df['roberta_compound'] < 0]['reviewText'].str.cat(sep=' ')

# Generate WordCloud for positive reviews
wordcloud_positive = WordCloud(width=400, height=200, background_color='white', colormap='Greens', stopwords=ENGLISH_STOP_WORDS).generate(positive_reviews)

# Generate WordCloud for negative reviews
wordcloud_negative = WordCloud(width=400, height=200, background_color='white',  colormap='Reds', stopwords=ENGLISH_STOP_WORDS).generate(negative_reviews)

# Plot the combined WordCloud
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.title('Word Cloud - Positive Reviews')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.title('Word Cloud - Negative Reviews')
plt.axis('off')

# Add a legend
plt.figlegend(labels=['Positive Reviews', 'Negative Reviews'], loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize='large')

plt.tight_layout()
plt.show()

#-----------------------------------------------------------------------------------------------------

#Checking if for the individual product types, if there is any significant relation 
#between its price-sentiment scores and price-ratings

bf=final_df.groupby('product type').agg(Counts=('product type', 'size')).reset_index() #making a list of the product types
product = bf["product type"].tolist() #creating a list of the 42 products

#creating a dictionary to save correlations
result={}

#looping through each of the 42 product catagories
for cat in product:
    df= final_df[final_df["product type"] == cat] #creating a df on that each product type
    cor= df['price'].corr(df['roberta_compound']) #correlation between its price and sentimentscore
    myid= cat #using product type name as id
    result[myid] = cor #saving correlations by each product type in the dictionary
    if cor> 0.3 or cor< -0.3:
        print(cat, cor) #printing if any of those correlations are more 0.3 or less than -0.3

#Only Single Wall Oven is printed with 0.46 correlation between its price and sentiment score
#moderately positive correlation

gf=final_df[final_df["product type"] == 'Single Wall Oven'] #making a df with only Single Wall Oven
len(gf) #Only 13 rows with this product

#Single Wall Oven's Sentiment Score vs Price
plt.figure(figsize=(8, 6))
sns.barplot(data=gf, x='price', y='roberta_compound')
plt.title('Senitment Score by Price for Single Wall Oven')
plt.show()

#Price is not significantly related to the sentiment of the reviews 

#following the same procedure to find correlations between price and ratings of each product type
output={}
for cat in product:
    df= final_df[final_df["product type"] == cat]
    cor= df['price'].corr(df['ratings'])
    myid= cat
    output[myid] = cor
    if cor> 0.3 or cor< -0.3:
        print(cat, cor)

#Only Single Wall Oven is printed with 0.50 correlation between its price and ratings
#moderately positive correlation

#Single Wall Oven's Sentiment Score vs Price
plt.figure(figsize=(8, 6))
sns.barplot(data=gf, x='ratings', y='price')
plt.title('Price by Ratings for Single Wall Oven')
plt.show()

#Price is not significantly related to the ratings of the reviews

#-----------------------------------------------------------------------------------------------------

#Check if any specific product type has high negative reviews than the overall dataset

#Making dataframe with postisive and negative reviews
positive_reviews = final_df[final_df['roberta_compound'] > 0]
negative_reviews = final_df[final_df['roberta_compound'] < 0]
#Percentage of Positive reviews
len(positive_reviews)/len(final_df) #77%
#Percentage of Negative reviews
len(negative_reviews)/len(final_df) #23%

#Lets fix our threshold 34% or more than 1/3 of the data
#If the percentage of Negative Reviews for any product type is more than threshold
#we will consider that as having high negative reviews

negativereview_percentage={} #creating a dictionary to save the percent of negative reviews

#looping through each of the 42 product catagories
for cat in product:
    product_df= final_df[final_df["product type"] == cat] #creating a df on that each product type
    negative_df=product_df[product_df['roberta_compound'] < 0] #creatin a df on its negative reviews
    percent=len(negative_df)/len(product_df) #percentage of the product type's negative reviews
    myid= cat #using product type name as id
    negativereview_percentage[myid] = percent #saving percentage of negative reviews by each product type in the dictionary
    if percent> 0.34:
        print(cat,'has a negative reviews of', round(percent*100),'% out of', len(product_df), 'reviews') #printing if product type whose negative reviews are more than 1/3 of the data

#Out of the 42 product types, 
#Built in dishwasher has a negative reviews of 39%, but it has only 23 rows (lacking enough size)
# Dryer Vents has a negative reviews of 35%, and it has 69 rows
# Range Knobs has a negative reviews of 39%, and it has 500 rows

three=['Dryer Vents','Range Knobs']
       #make a list of these two products, who has enough size and significant % of negative reviews

for cat in three: #looping through these products from the list
    gf=final_df[final_df["product type"] == cat] #making a df with each product type
    #Vizualize the RoBEERTa Score Distribution for these products
    plt.figure(figsize=(6, 4))
    plt.hist(gf['roberta_compound'], bins='auto', edgecolor='black')  # 'auto' determines the number of bins automatically
    plt.title(f'Histogram of RoBERTa Sentiment Score of {cat}')
    plt.xlabel('RoBERTa Sentiment score')
    plt.ylabel('Frequency')
    plt.show()

#Dryer Vents are having some neutrally negative reviews
#So we can report that Range Knobs is having significant negative reviews