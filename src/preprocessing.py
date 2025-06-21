# preprocessing.py
import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
import os

# Download stopwords once
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to clean a single text entry
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Function to load and preprocess the dataset
def load_and_preprocess(input_path, output_path=None):
    df = pd.read_csv(input_path)
    df.columns = ['sms', 'label']  # Rename columns if necessary
    df['clean_sms'] = df['sms'].apply(preprocess_text)

    # Save if output path is provided
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"[INFO] Preprocessed data saved to: {output_path}")

    return df
