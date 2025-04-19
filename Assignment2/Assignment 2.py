# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 19:56:53 2025

@author: leor7
"""

import os
import json
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
import re

# File paths
base_dir = r"C:\NLP\Assignments\Assignment 2"
os.makedirs(base_dir, exist_ok=True)
file_path = os.path.join(base_dir, "leor_intents.json")

# Load intents data
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Initialize lists for data storage
patterns, tags = [], []
responses = {}

# Extract data from JSON
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])
    responses[intent["tag"]] = intent["responses"]

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    return text

# Apply text cleaning to patterns
patterns = [clean_text(p) for p in patterns]

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(tags)

# Save label encoder
with open(os.path.join(base_dir, "label_encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)

# Tokenize text patterns
vocab_size = 900
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(patterns)

# Save tokenizer
with open(os.path.join(base_dir, "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)

# Convert patterns to sequences
patterns_sequences = tokenizer.texts_to_sequences(patterns)

# Pad sequences to max length of 40
max_sequence_length = 40
patterns_padded = pad_sequences(patterns_sequences, maxlen=max_sequence_length, padding='post')

# Define deep learning model
embedding_dim = 20
num_classes = len(set(tags))

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(10, activation='sigmoid'),
    Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model summary
model.summary()

# Train model
print("\nTraining model with 500 epochs...")
model.fit(patterns_padded, np.array(labels), epochs=500, verbose=1)

print("\nTraining model with 1000 epochs...")
history_1000 = model.fit(patterns_padded, np.array(labels), epochs=1000, verbose=1)

# Save only the 1000-epoch trained model
model.save(os.path.join(base_dir, "saved_model.keras"))  # Recommended format
model.save(os.path.join(base_dir, "saved_model.h5"))  # Alternative HDF5 format

print("\nModel, tokenizer, and label encoder saved successfully!")
