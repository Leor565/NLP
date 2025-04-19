# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 12:14:13 2025

@author: leor7
"""

import json
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Load the JSON file
with open(r"C:\NLP\Assignments\Assignment 3\recipies.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# a. Total number of recipes
total_recipes = len(data)
print(f"Total number of recipes: {total_recipes}")

# b. Number of unique cuisines
cuisine_set = {recipe.get("cuisine") for recipe in data if "cuisine" in recipe}
print(f"Number of unique cuisines: {len(cuisine_set)}")

# List all cuisines
print("\nList of cuisines:")
for cuisine in sorted(cuisine_set):
    print(f" - {cuisine}")

# c. Cuisine summary with recipe counts
print("\nCuisine-wise recipe count:")
cuisine_counts = {}
for recipe in data:
    cuisine = recipe.get("cuisine")
    if cuisine:
        cuisine_counts[cuisine] = cuisine_counts.get(cuisine, 0) + 1

# Print in sorted order
for cuisine in sorted(cuisine_counts):
    print(f" - {cuisine}: {cuisine_counts[cuisine]} recipes")

# Convert cuisine set to lowercase for comparison
available_cuisines = {c.lower() for c in cuisine_set}

# Start interactive loop
while True:
    cuisine_input = input("\nEnter a cuisine type (or type 'exit' to quit): ").strip().lower()
    
    if cuisine_input == "exit":
        print("Goodbye!")
        break

    if cuisine_input not in available_cuisines:
        print(f"We don’t have recommendations for {cuisine_input.capitalize()}. Please try another cuisine.")
        continue

    # Filter recipes of the selected cuisine
    filtered_recipes = [item["ingredients"] for item in data if item["cuisine"].lower() == cuisine_input]
    num_cuisine_recipes = len(filtered_recipes)
    print(f"\nNumber of recipes for '{cuisine_input.capitalize()}': {num_cuisine_recipes}")

    # Prepare data for Apriori
    te = TransactionEncoder()
    te_array = te.fit(filtered_recipes).transform(filtered_recipes)
    df = pd.DataFrame(te_array, columns=te.columns_)

    # Apriori algorithm parameters
    min_support = 100 / num_cuisine_recipes
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

    # Show top frequent itemset
    if not frequent_itemsets.empty:
        top_itemset = frequent_itemsets.sort_values(by="support", ascending=False).iloc[0]
        print("\nTop frequent ingredient group:")
        print(f"{list(top_itemset['itemsets'])} (support: {top_itemset['support']:.3f})")
    else:
        print("\nNo frequent itemsets found.")

    # Show rules with lift > 2
    strong_rules = rules[rules['lift'] > 2]

    if not strong_rules.empty:
        print("\nAssociation rules with lift > 2:")
        for _, row in strong_rules.iterrows():
            print(f"Rule: {set(row['antecedents'])} -> {set(row['consequents'])} | "
                  f"conf: {row['confidence']:.2f}, lift: {row['lift']:.2f}")
    else:
        print("\nNo association rules found with lift > 2.")
