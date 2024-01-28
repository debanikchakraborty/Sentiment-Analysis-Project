import numpy as np
import pandas as pd
from google.cloud import language
from google.oauth2 import service_account

#Note:
#'sample_appliance.csv' file was our sample file where we did the manual Human Sentiment Scoring 
#and recorded the True Review Type.
#We renamed the file as 'sample_appliance complete.csv' and loaded it for further analysis


# Task 1: Read the file into a data frame
file_path = "C:/Users/atanu/OneDrive - Oklahoma A and M System/SEMESTER1/MSIS5193/PythonGroupProject/Data/sample_appliance complete.csv"
df = pd.read_csv(file_path, keep_default_na=False)
# Replace NaN values with empty strings
df.replace(pd.NA, '', inplace=True)


# Task 2: Call Google API to get sentiment score and magnitude
def analyze_text_sentiment(text):
    if text is not None:
        text = str(text)
        # Load the key file
        creds = service_account.Credentials.from_service_account_file(
            'C:/Users/atanu/OneDrive - Oklahoma A and M System/SEMESTER1/MSIS5193/Ass8/temporal-grin-406201-7197b23ea06e.json')
        # Initialize a client using the key file
        client = language.LanguageServiceClient(credentials=creds)
        # Convert the text into a document type
        document = language.Document(content=text, type_=language.Document.Type.PLAIN_TEXT)

        # Call function to analyze the sentiment of the text
        response = client.analyze_sentiment(document=document)

        # Get the results in the variable sentiment
        sentiment = response.document_sentiment

        # Return the sentiment score and magnitude
        return pd.Series({'Sentiment_Score': sentiment.score, 'Sentiment_Magnitude': sentiment.magnitude})
    else:
        return pd.Series({'Sentiment_Score': None, 'Sentiment_Magnitude': None})


# Task 3: Add two columns to the file “SampleAppliance.csv” and save the results into a new file as “SampleAppliance_Output.csv”
df[['Sentiment_Score', 'Sentiment_Magnitude']] = df['reviewText'].apply(analyze_text_sentiment).apply(pd.Series).round(2)
# print(df.head())
df['Normalized_Sentiment_Scores'] = np.tanh(df['Sentiment_Score'] / df['Sentiment_Magnitude']).round(2)
output_file_path = "C:/Users/atanu/OneDrive - Oklahoma A and M System/SEMESTER1/MSIS5193/PythonGroupProject/Data/SampleAppliance_Output.csv"
df.to_csv(output_file_path, index=False, na_rep='')

# Display the normalized sentiment scores
# print(df['Normalized_Sentiment_Scores'])
