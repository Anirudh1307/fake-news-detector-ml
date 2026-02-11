import pandas as pd
import joblib
import re
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

FAKE_NEWS_PATH = 'data/Fake.csv'
TRUE_NEWS_PATH = 'data/True.csv'
MODEL_FILENAME = 'logistic_regression_model.pkl'
VECTORIZER_FILENAME = 'tfidf_vectorizer.pkl'

app = Flask(__name__)
model = None
vectorizer = None

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def load_artifacts():
    global model, vectorizer
    try:
        model = joblib.load(MODEL_FILENAME)
        vectorizer = joblib.load(VECTORIZER_FILENAME)
        print(f"Successfully loaded Vectorizer from {VECTORIZER_FILENAME}")
        print(f"Successfully loaded Model from {MODEL_FILENAME}")
    except FileNotFoundError as e:
        print(f"[ERROR] Model or Vectorizer file not found: {e}. Run train_model.py first.")
        model = None
        vectorizer = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model not loaded. Please ensure model files exist.'}), 500

    data = request.get_json()
    text = data.get('text', '')

    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    
    prediction = model.predict(vectorized_text)[0]
    confidence_score = model.predict_proba(vectorized_text)[0]
    
    # Determine predicted class and confidence
    if prediction == 0:
        label = "FAKE NEWS"
        confidence = confidence_score[0] * 100
    else:
        label = "REAL NEWS"
        confidence = confidence_score[1] * 100

    # Feature Contribution (Top Words)
    # Get the feature names (words)
    feature_names = vectorizer.get_feature_names_out()
    
    # Get the coefficients (weights) from the model
    # For binary Logistic Regression, model.coef_[0] contains the weights for class 1 (Real News)
    # Negative weights push towards class 0 (Fake News)
    coefficients = model.coef_[0]
    
    # Get the indices of the highest positive and negative coefficients
    # Positive weights (Real News)
    real_indices = coefficients.argsort()[-10:][::-1]
    # Negative weights (Fake News)
    fake_indices = coefficients.argsort()[:10]

    # Map indices to words and store as list of (word, weight)
    real_contributors = []
    for i in real_indices:
        real_contributors.append({'word': feature_names[i], 'weight': float(coefficients[i])})

    fake_contributors = []
    for i in fake_indices:
        fake_contributors.append({'word': feature_names[i], 'weight': float(coefficients[i])})

    return jsonify({
        'prediction': label,
        'confidence': f"{confidence:.2f}%",
        'real_contributors': real_contributors,
        'fake_contributors': fake_contributors
    })

if __name__ == '__main__':
    load_artifacts()
    # Explicitly set host to '0.0.0.0' to fix connectivity issues
    app.run(debug=True, host='0.0.0.0')
