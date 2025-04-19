# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 20:41:43 2025

@author: leor7
"""

import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings 

leor_df = pd.read_csv(r'C:\NLP\Assignments\Assignment 1\COVID19_mini.csv')

print(leor_df.head())

leor_df.drop("user", axis=1, inplace=True)

print(leor_df.head())

import re

def clean_tweet(tweet): 
    # Example: Remove 'RT' at the beginning and anything after the tweet, like URLs, mentions, etc.
   tweet = re.sub(r"^RT\s?@\w+: ", "", tweet)  # Remove 'RT @user:' at the start of the tweet
   tweet = re.sub(r"@\w+", "", tweet)  # Remove mentions (@user)
   tweet = re.sub(r"http\S+", "", tweet)  # Remove URLs
   tweet = re.sub(r"\s+", " ", tweet)  # Remove extra spaces
   tweet = tweet.strip()  # Remove leading/trailing whitespaces
   return tweet

# Apply the cleaning function to the 'text' column
leor_df['text'] = leor_df['text'].apply(clean_tweet)

# Show the cleaned data
print(leor_df.head())

# Check for null values in the text column
print(leor_df['text'].isnull().sum())

# Check and drop duplicate tweets
leor_df.drop_duplicates(subset=['text'], inplace=True)

# Convert all text to lowercase
leor_df['text'] = leor_df['text'].apply(lambda x: x.lower())
# Remove special characters except for spaces
leor_df['text'] = leor_df['text'].apply(lambda x: re.sub(r'[^A-Za-z0-9\s]+', '', x))

print(leor_df)


import nlpaug.augmenter.word as naw


import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
leor_df = pd.read_csv(r'C:\NLP\Assignments\Assignment 1\COVID19_mini.csv')

# Function to remove stop words from a tweet
def remove_stop_words(tweet):
    stop_words = set(stopwords.words('english'))  # Load English stopwords
    tokens = word_tokenize(tweet)  # Tokenize the tweet into words
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]  # Filter out stop words
    return ' '.join(filtered_tokens)  # Reassemble the tweet without stop words

# Function to apply synonym substitution using Word Embeddings
def substitute_with_synonyms(tweet, num_insertions=2):
    # Use GloVe embeddings for substitution 
    augmenter = naw.WordEmbsAug(
        action="substitute",  
        model_type="glove",   
        model_path=r"C:\NLP\week 3\glove.6B\glove.6B.100d.txt",  
        top_k=5  # Get top 5 similar words
    )
    
  
    augmented_text = augmenter.augment(tweet)
    
    return augmented_text  

# Apply stop word removal to each tweet in the dataset
leor_df['cleaned_text'] = leor_df['text'].apply(lambda x: remove_stop_words(x))

# Apply the synonym substitution to each cleaned tweet
augmented_tweets = []
for _, row in leor_df.iterrows():
    original_text = row['cleaned_text']
    augmented_text = substitute_with_synonyms(original_text, num_insertions=2)
    augmented_tweets.append({
        'original_text': original_text,
        'augmented_text': augmented_text,
        'sentiment': row['sentiment']  # Keep the original sentiment
})

# Create a new dataframe with augmented tweets
augmented_df = pd.DataFrame(augmented_tweets)

# Export the augmented dataset to a text file
augmented_df.to_csv('leor_df_after_random_insertion.csv', index=False, sep='\t')

# Show the augmented dataframe (for verification)
print(augmented_df.head())
