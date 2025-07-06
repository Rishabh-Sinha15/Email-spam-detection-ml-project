import os
from flask import Flask, request, jsonify
from preprocessing import preprocess_text
from train import train_model

import joblib

# Locate train.csv (go one level up from src/)
BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'train.csv')

# Fail fast if train.csv not found
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"train.csv not found at {DATA_PATH}")
print(f"[INFO] Loading training data from: {DATA_PATH}")

# Train model on API startup
print("[INFO] Training model from train.py on API startup...")
model, vectorizer, _, _ = train_model(data_path=DATA_PATH)

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "✅ API is running. Use POST /predict with JSON: {'sms': 'your message here'}"

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json(silent=True)
    if not data or 'sms' not in data:
        return jsonify(error="Missing 'sms' field in JSON body"), 400

    raw_text = data['sms']
    try:
        clean = preprocess_text(raw_text)
        x_vec = vectorizer.transform([clean])
        prediction = int(model.predict(x_vec)[0])
    except Exception as e:
        return jsonify(error=str(e)), 500

    return jsonify(label=prediction), 200

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
