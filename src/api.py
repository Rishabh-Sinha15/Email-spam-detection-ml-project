# api.py

import os
from flask import Flask, request, jsonify
from preprocessing import load_and_preprocess, preprocess_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Locate train.csv
# 1) Check current working directory
if os.path.exists('train.csv'):
    DATA_PATH = os.path.abspath('train.csv')
else:
    # 2) Fallback: one level up from this file (src/) → project root
    BASE_DIR    = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
    DATA_PATH   = os.path.join(PROJECT_ROOT, 'train.csv')

# Fail fast if CSV not found
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"train.csv not found at {DATA_PATH}")

print(f"[INFO] Loading training data from: {DATA_PATH}")

# 1) Load & clean
df = load_and_preprocess(DATA_PATH)
X = df['clean_sms']
y = df['label']

# 2) Train vectorizer & model
vectorizer = TfidfVectorizer()
X_vec      = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)
model = LogisticRegression()
model.fit(X_train, y_train)

# 3) Prediction helper
def predict_sms(raw_text: str) -> int:
    """
    1) Preprocess the raw text to match training cleanup
    2) Vectorize it
    3) Return 0 (not spam) or 1 (spam)
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("Empty or invalid SMS text provided.")

    # Clean exactly as during training
    clean = preprocess_text(raw_text)
    x_vec = vectorizer.transform([clean])
    return int(model.predict(x_vec)[0])

# 4) Flask app 
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to Team Number 4 Flask App"

@app.route('/predict', methods=['POST'])
def predict_route():
    # Parse JSON body
    data = request.get_json(silent=True)
    if not data or 'sms' not in data:
        return jsonify(error="Missing 'sms' field in JSON body"), 400

    raw_text = data['sms']
    try:
        label = predict_sms(raw_text)
    except Exception as e:
        return jsonify(error=str(e)), 500

    return jsonify(label=label), 200

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
