import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

def train_and_save_model():
    # Load local CSV files
    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")

    # Add labels: fake = 1, real = 0
    fake["label"] = 1
    true["label"] = 0

    # Combine datasets
    df = pd.concat([fake, true])
    df = df[["text", "label"]].dropna()

    # Train-test split
    X = df["text"]
    y = df["label"]

    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_vec, y)

    # Save model
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/fake_news_model.pkl")
    joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

    print("✅ Model and vectorizer saved to 'model/' folder.")

if __name__ == "__main__":
    train_and_save_model()
