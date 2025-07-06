from preprocessing import load_and_preprocess
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd

# ✅ Modified to accept data_path
def train_model(C=1.0, solver='lbfgs', max_iter=1000, data_path='train.csv'):
    df = load_and_preprocess(data_path)
    X = df['clean_sms']
    y = df['label']

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    model = LogisticRegression(C=C, solver=solver, max_iter=max_iter)
    model.fit(X_train, y_train)

    return model, vectorizer, X_test, y_test
