# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 09:40:21 2025

@author: leor
"""

#Step 1: Load the Dataset 
import pandas as pd
import gzip
import json
import numpy as np

def parse_json_lines(path):
    data = []
    with gzip.open(path, 'rb') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Load directly from the .gz file
file_path = "meta_Digital_Music.json.gz"
raw_data = parse_json_lines(file_path)

# Convert to DataFrame
songs_Leor = pd.DataFrame(raw_data)

# Normalize column names (lowercase)
songs_Leor.columns = songs_Leor.columns.str.lower()

print("Shape of dataset:", songs_Leor.shape)
print("\nFirst few rows:")
print(songs_Leor[['title', 'description']].head())

#Step 2: Data Exploration 

# Check for null values
print("\nNull values per column:")
print(songs_Leor.isnull().sum())

# Check for empty or invalid descriptions
def is_empty_description(desc):
    return desc == [] or desc == "" or desc is None or (isinstance(desc, str) and desc.strip() == "")

empty_descriptions = songs_Leor['description'].apply(is_empty_description).sum()
print(f"\nNumber of empty descriptions: {empty_descriptions}")

# Unique song titles
print("Unique titles:", songs_Leor['title'].nunique())

# Print column names
print("\nAll column names:")
print(songs_Leor.columns.tolist())

# Step 3(a): Clean & Prepare Data 

# Keep only needed columns
songs_clean = songs_Leor[['title', 'description']].copy()

# Drop rows with empty or invalid descriptions
songs_clean = songs_clean[~songs_clean['description'].apply(is_empty_description)].reset_index(drop=True)

# Convert description to lowercase joined string (if it's a list)
def format_description(desc):
    if isinstance(desc, list):
        return " ".join(desc).lower()
    elif isinstance(desc, str):
        return desc.lower()
    else:
        return ""

songs_clean['description'] = songs_clean['description'].apply(format_description)

#Step 3(b): TF-IDF Vectorization 

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(songs_clean['description'])

print("TF-IDF matrix shape:", tfidf_matrix.shape)

#Step 3(c): Compute Cosine Similarity 

from sklearn.metrics.pairwise import cosine_similarity

cosine_sim_matrix = cosine_similarity(tfidf_matrix)
print("Cosine similarity matrix shape:", cosine_sim_matrix.shape)

# Step 3(d): Store Top 10 Recommendations

title_list = songs_clean['title'].tolist()
recommendations_dict = {}

for idx, title in enumerate(title_list):
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Exclude self

    recommended_titles = [title_list[i[0]] for i in sim_scores]
    recommendations_dict[title.lower()] = recommended_titles  # Store with lowercase key

# Save recommendations to a JSON file
with open("song_recommendations.json", "w", encoding="utf-8") as f:
    json.dump(recommendations_dict, f, indent=2)

# Step 4: Recommender Function & Input Loop 

# Load recommendations
with open("song_recommendations.json", "r", encoding="utf-8") as f:
    recommendations = json.load(f)

# Input loop
while True:
    user_input = input("\nEnter a song title (or 'exit' to quit): ").strip()

    if user_input.lower() == "exit":
        print("Exiting the recommender.")
        break

    key = user_input.lower()

    if key not in recommendations:
        print(f"We don’t have recommendations for '{user_input}'. Please try another song.")
        continue

    print(f"\nTop 10 recommended songs similar to '{user_input}':")
    for idx, title in enumerate(recommendations[key], 1):
        print(f"{idx}. {title}")
