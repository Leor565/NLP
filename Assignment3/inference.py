# -*- coding: utf-8 -*-
"""
Inference Script for Song Recommender
Loads recommendations from JSON and interacts with user
"""

import json

# -------------------- Load Recommendations --------------------
try:
    with open(r"C:\NLP\Assignments\Assignment 3\song_recommendations.json", "r", encoding="utf-8") as f:
        recommendations = json.load(f)
    print("Recommendations loaded successfully.")
except FileNotFoundError:
    print("Error: 'song_recommendations.json' not found.")
    exit()

# -------------------- Input Loop --------------------
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
