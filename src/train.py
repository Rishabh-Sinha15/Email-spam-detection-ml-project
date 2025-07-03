# train.py
from preprocessing import load_and_preprocess
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd

# 1) Load & clean
df = load_and_preprocess('train.csv')
# IMPORTANT: if you created a "clean_sms" column, use that, otherwise SMS may still have punctuation/stopwords
X = df['clean_sms']    # ← here, not df['sms']
y = df['label']

# 2) Train vectorizer & model
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# 3) Prediction helper
def predict_sms(raw_text: str):
    # Apply the SAME cleaning + vectorizing as training
    # If your API is passing cleaned text, skip the preprocess_text step here.
    x_vec = vectorizer.transform([raw_text])   # shape (1, n_features)
    pred = model.predict(x_vec)                # returns array length 1
    return pred[0]
