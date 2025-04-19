# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 12:11:38 2025

@author: leor7
"""

import json
import random
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
model_path = r"C:\NLP\Assignments\Assignment 2\saved_model.keras"
model = load_model(model_path)

# Load tokenizer and label encoder
tokenizer_path = r"C:\NLP\Assignments\Assignment 2\tokenizer.pkl"
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

encoder_path = r"C:\NLP\Assignments\Assignment 2\label_encoder.pkl"
with open(encoder_path, "rb") as f:
    label_encoder = pickle.load(f)

# Load intents JSON
intents_path = r"C:\NLP\Assignments\Assignment 2\leor_intents.json"
with open(intents_path, "r", encoding="utf-8") as f:
    intents = json.load(f)

# Helper: get response from predicted tag
def get_response(tag, intents_json):
    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Sorry, I didn't understand that."

# Chat loop
print("Chatbot is running! Type 'bye' to exit.")
while True:
    user_input = input("You: ").strip().lower()
    if user_input == "bye":
        print("Chatbot: Goodbye!")
        break

    # Tokenize and pad the input
    sequence = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(sequence, maxlen=40, padding="post", truncating="post")

    # Predict intent
    prediction = model.predict(padded, verbose=0)
    predicted_index = np.argmax(prediction)
    predicted_tag = label_encoder.inverse_transform([predicted_index])[0]

    # Get response based on the predicted tag
    response = get_response(predicted_tag, intents)
    print(f"Chatbot: {response}")
