# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 19:43:29 2025

@author: leor7
"""

import nltk
import pandas as pd 
from  sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score
import matplotlib.pyplot as plt 
import re 

df = pd.read_csv(r"C:\NLP\Assignments\Assignment 1\COVID19_data.csv")


print(df.shape)
print(df.head())
print(df.dtypes)  

df = df.drop("user", axis= 1)


# Function to clean each tweet
def clean_tweet(tweet):
    tweet = re.sub(r"^\s*@\w+\s*", "", tweet)  # Remove leading mentions (e.g., "@user123 ")
    tweet = re.sub(r"https?://\S+", "", tweet)  # Remove URLs
    tweet = re.sub(r"\s*#\w+\s*$", "", tweet)  # Remove trailing hashtags (e.g., " #COVID19")
    tweet = tweet.strip()  # Remove extra spaces
    return tweet

# Apply cleaning to the "tweet" column
df["text"] = df["text"].astype(str).apply(clean_tweet)

# Display cleaned tweets
print(df["text"].head())


# Add a new column with the length of each cleaned tweet
df["tweet_length"] = df["text"].apply(len)

# Display the first few rows
print(df[["text", "tweet_length"]].head())


# Load positive and negative words with explicit encoding
with open(r"C:\NLP\Assignments\Assignment 1\positive-words.txt", "r", encoding="utf-8", errors="ignore") as f:
    positive_words = set(f.read().splitlines())

with open(r"C:\NLP\Assignments\Assignment 1\negative-words.txt", "r", encoding="utf-8", errors="ignore") as f:
    negative_words = set(f.read().splitlines())


# Function to count positive and negative words and compute percentages
def count_sentiment_words(tweet):
    words = tweet.lower().split()  # Convert to lowercase and split into words
    total_words = len(words) if len(words) > 0 else 1  # Avoid division by zero
    
    pos_count = sum(1 for word in words if word in positive_words)
    neg_count = sum(1 for word in words if word in negative_words)
    
    pos_score = pos_count / total_words
    neg_score = neg_count / total_words
    
    pos_percentage = pos_score * 100
    neg_percentage = neg_score * 100
    
    return pd.Series([pos_score, neg_score, pos_percentage, neg_percentage])

# Apply the function to calculate sentiment scores and percentages
df[["positive_score", "negative_score", "positive_percentage", "negative_percentage"]] = df["text"].apply(count_sentiment_words)

# Display first few rows
print(df[["text", "positive_score", "negative_score", "positive_percentage", "negative_percentage"]].head())
print(df.dtypes)  

# Function to assign sentiment labels based on sentiment percentages
def assign_sentiment_label(row):
    if (row["positive_percentage"] == 0 and row["negative_percentage"] == 0) or \
       (row["positive_percentage"] == row["negative_percentage"]):
        return "neutral"
    elif row["positive_percentage"] > row["negative_percentage"]:
        return "positive"
    else:
        return "negative"

# Apply function to create a new column
df["predicted_sentiment_score"] = df.apply(assign_sentiment_label, axis=1)

# Display first few rows with the new sentiment column
print(df[["text", "positive_percentage", "negative_percentage", "predicted_sentiment_score"]].head())
print(df.dtypes)

from sklearn.metrics import f1_score, accuracy_score

# Check if the 'sentiment' column exists
if "sentiment" in df.columns:
    # Map categorical labels to numerical values
    label_mapping = {"positive": 1, "negative": 0, "neutral": 2}
    y_true = df["sentiment"].map(label_mapping)
    y_pred = df["predicted_sentiment_score"].map(label_mapping)

    # Compute F1 score and accuracy
    f1 = f1_score(y_true, y_pred, average="weighted")  # Change "weighted" to "macro" if needed
    accuracy = accuracy_score(y_true, y_pred)

    # Print results
    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
else:
    print("Error: 'sentiment' column not found in the dataset.")

