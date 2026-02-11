import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

FAKE_NEWS_PATH = 'data/Fake.csv'
TRUE_NEWS_PATH = 'data/True.csv'
MODEL_FILENAME = 'logistic_regression_model.pkl'
VECTORIZER_FILENAME = 'tfidf_vectorizer.pkl'

def preprocess_text(text):
    if isinstance(text, str):
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.lower()
    return ""

def load_and_prepare_data():
    try:
        df_fake = pd.read_csv(FAKE_NEWS_PATH)
        df_true = pd.read_csv(TRUE_NEWS_PATH)

        df_fake['label'] = 0
        df_true['label'] = 1

        df = pd.concat([df_fake, df_true], ignore_index=True)

        df['content'] = df['title'] + ' ' + df['text']

        df['content'] = df['content'].apply(preprocess_text)

        df.replace('', np.nan, inplace=True)
        df.dropna(subset=['content'], inplace=True)

        return df
    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {e.filename}. Please ensure data files are located in the 'data/' directory.")
        return None
    except Exception as e:
        print(f"[ERROR] An error occurred during data loading and preparation: {e}")
        return None

def train_and_save_model(df):
    print("Starting model training process...")

    X = df['content']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_vec = tfidf_vectorizer.fit_transform(X_train)
    X_test_vec = tfidf_vectorizer.transform(X_test)

    print(f"Vectorizer fitted. Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")

    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained successfully. Test Accuracy: {accuracy:.4f}")

    joblib.dump(model, MODEL_FILENAME)
    joblib.dump(tfidf_vectorizer, VECTORIZER_FILENAME)
    print(f"Model saved to {MODEL_FILENAME}")
    print(f"Vectorizer saved to {VECTORIZER_FILENAME}")

if __name__ == "__main__":
    combined_df = load_and_prepare_data()

    if combined_df is not None:
        train_and_save_model(combined_df)
