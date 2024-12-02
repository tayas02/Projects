# Import required libraries
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
import re
import os

# Install pypdf if not already installed


# Load GatorTron model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("UFNLP/gatortron-large")
model = AutoModel.from_pretrained("UFNLP/gatortron-large")

# Parse PDF document and split into sentences
def parse_document(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + " "
    sentences = re.split(r'(?<=[.!?]) +', text)
    return sentences

# Get CLS token embedding for each sentence
def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
    return cls_embedding.squeeze().numpy()

# Read PDF and generate embeddings for each sentence
file_path = "/content/sample.pdf"  # Replace with your PDF file path
sentences = parse_document(file_path)
embeddings = np.array([get_sentence_embedding(sentence) for sentence in sentences])

# Process query and get its embedding
query = "Is the patient experiencing any residual back pain, and how is it managed?"
query_inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512)
with torch.no_grad():
    query_outputs = model(**query_inputs)
query_embedding = query_outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# Calculate cosine similarities between query and each sentence embedding
similarities = cosine_similarity([query_embedding], embeddings)[0]

# Define minimum and maximum thresholds
initial_threshold = 0.5
max_threshold = 10
step = 0.01  # Increment step for threshold adjustment

# Initialize variables to store the best response and threshold
best_response = None
best_threshold = initial_threshold

threshold = initial_threshold
while threshold <= max_threshold:
    response = None  # Reset response for each threshold

    # Check if there's any sentence with similarity above the current threshold
    for i, sim in enumerate(similarities):
        if sim > threshold:
            response = f"{sentences[i]}"
            break  # Found a response, so break the inner loop

    # If a response was found, store it as the current best and increase the threshold
    if response:
        best_response = response
        best_threshold = threshold
        threshold += step  # Try the next higher threshold
    else:
        # No response found at the current threshold, stop the loop
        break

# Output the best result found at the highest valid threshold
if best_response:
    print(f"Best response: '{best_response}' at threshold: {best_threshold}")
else:
    print("No similar sentences found.")
