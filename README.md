# Fake News Detector (Production-Grade AI Portfolio Project)

End-to-end Flask AI system for fake-news style classification with:

- Advanced NLP preprocessing
- Multi-model training and automatic model selection
- SHAP + LIME explainability
- URL-based article analysis (`/analyze_url`)
- Analytics dashboard
- Unit tests and Docker deployment

## 1) Updated Project Architecture

```text
fake-news-detector/
|
|-- app/
|   |-- __init__.py
|   |-- routes.py
|   |-- model_loader.py
|   |-- preprocessing.py
|   |-- explainability.py
|   `-- utils.py
|
|-- training/
|   |-- __init__.py
|   |-- dataset_loader.py
|   |-- train_models.py
|   |-- evaluate_models.py
|   `-- train_transformer.py        (optional DistilBERT)
|
|-- dashboard/
|   |-- __init__.py
|   `-- analytics.py
|
|-- templates/
|   |-- index.html
|   `-- dashboard.html
|
|-- tests/
|   |-- conftest.py
|   |-- test_preprocessing.py
|   |-- test_prediction.py
|   `-- test_api.py
|
|-- models/                         (generated artifacts)
|   `-- reports/                    (generated evaluation charts/tables)
|
|-- data/
|   |-- Fake.csv / True.csv         (training input)
|   `-- train.tsv/valid.tsv/test.tsv (optional LIAR source files)
|
|-- app.py                          (Flask entrypoint)
|-- train_model.py                  (compat wrapper -> training/train_models.py)
|-- prepare_liar_data.py
|-- requirements.txt
|-- requirements-transformer.txt
|-- Dockerfile
`-- README.md
```

## 2) Advanced NLP Pipeline

Implemented in [`app/preprocessing.py`](app/preprocessing.py):

- Lowercasing
- URL removal
- Punctuation removal
- Non-alphabetic cleanup
- Stopword removal
- Lemmatization (WordNet if available, fallback rule-based)
- Duplicate text removal (dataset-level)

Core functions:

- `preprocess_text(...)`
- `preprocess_corpus(...)`
- `deduplicate_texts(...)`
- `deduplicate_dataframe(...)`

## 3) Multi-Model Training + Auto Selection

Implemented in [`training/train_models.py`](training/train_models.py):

Models trained:

- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM
- Random Forest

Metrics used for comparison:

- Accuracy
- Precision
- Recall
- F1-score

Best model is selected automatically (ranked primarily by F1-score), then saved to:

- `models/best_model.joblib`
- `models/tfidf_vectorizer.joblib`
- `models/metadata.joblib`

## 4) Model Evaluation Reports

Implemented in [`training/evaluate_models.py`](training/evaluate_models.py).

Generated artifacts in `models/reports/`:

- Confusion matrix plot per model
- Classification report CSV per model
- ROC curve plot per model
- Accuracy comparison table (`accuracy_comparison.csv`)
- Accuracy comparison plot (`accuracy_comparison.png`)

## 5) Explainable AI (SHAP + LIME)

Implemented in [`app/explainability.py`](app/explainability.py):

- Global indicative words (model-level)
- Local indicative words (input-level)
- SHAP explanation payload
- LIME explanation payload

API returns:

- `top_fake_words`
- `top_real_words`
- `explanation.shap`
- `explanation.lime`

If SHAP/LIME is unavailable in environment, API returns graceful fallback reasons.

## 6) Flask API (Updated)

Implemented in [`app/routes.py`](app/routes.py).

### `POST /predict`

Request:

```json
{
  "text": "Your claim or article text",
  "include_shap": true,
  "include_lime": true
}
```

Response example:

```json
{
  "prediction": "FAKE NEWS",
  "confidence": 74.21,
  "top_fake_words": [{"word":"hoax","weight":-1.83}],
  "top_real_words": [{"word":"report","weight":1.24}],
  "explanation": {
    "top_fake_words": [...],
    "top_real_words": [...],
    "global_importance": {...},
    "shap": {"available": true, "items": [...]},
    "lime": {"available": true, "items": [...]}
  }
}
```

### `POST /analyze_url`

Fetches article text with `newspaper3k`, preprocesses it, predicts label, and returns explanations.

Request:

```json
{
  "url": "https://example.com/news-article",
  "include_shap": true,
  "include_lime": true
}
```

Response example:

```json
{
  "url": "https://example.com/news-article",
  "article_preview": "First 350 chars...",
  "article_char_count": 8421,
  "prediction": "REAL NEWS",
  "confidence": 81.34,
  "top_fake_words": [...],
  "top_real_words": [...],
  "explanation": {...}
}
```

### Additional endpoints

- `GET /health`
- `GET /dashboard`
- `GET /api/analytics`

## 7) Analytics Dashboard

Dashboard UI: [`templates/dashboard.html`](templates/dashboard.html)

Backend analytics summary: [`dashboard/analytics.py`](dashboard/analytics.py)

Dashboard shows:

- Number of analyzed items
- Fake vs real distribution
- Most common fake keywords
- Confidence distribution

Uses Chart.js for visualization.

## 8) Optional Transformer Support (DistilBERT)

Optional script: [`training/train_transformer.py`](training/train_transformer.py)

Install optional deps:

```bash
pip install -r requirements-transformer.txt
```

Then run:

```bash
python training/train_transformer.py
```

## 9) Unit Testing

Pytest suite under [`tests/`](tests):

- Preprocessing tests
- Model prediction tests
- API endpoint tests (`/predict`, `/analyze_url`, `/api/analytics`)

Run tests:

```bash
pytest -q
```

## 10) How to Run the Improved System

### Step A: Prepare data

If using LIAR TSV files:

```bash
python prepare_liar_data.py
```

This creates `data/Fake.csv` and `data/True.csv`.

### Step B: Install dependencies

```bash
pip install -r requirements.txt
```

### Step C: Train + compare models

```bash
python training/train_models.py
```

### Step D: Start Flask app

```bash
python app.py
```

Open:

- App: `http://127.0.0.1:5000`
- Dashboard: `http://127.0.0.1:5000/dashboard`

## Deployment

### Docker

Build:

```bash
docker build -t fake-news-detector .
```

Run:

```bash
docker run -p 5000:5000 fake-news-detector
```

### Render

1. Push repo to GitHub.
2. Create new Web Service in Render.
3. Commit required artifacts:
   - `models/best_model.joblib`
   - `models/tfidf_vectorizer.joblib`
4. Build command:
   - `pip install --no-cache-dir -r requirements.txt`
   - If your environment strips optional parsers, also ensure: `pip install newspaper3k`
5. Start command:
   - `gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 1 --timeout 120 app:app`
6. Set environment variables:
   - `ENABLE_SHAP=0`
   - `ENABLE_LIME=0`
   - `MAX_INPUT_CHARS=30000`
7. Add persistent disk if you want to preserve analytics logs.

### Railway

1. Create a new Railway project from GitHub repo.
2. Railway detects Dockerfile automatically (recommended).
3. If not using Docker:
   - Install command: `pip install --no-cache-dir -r requirements.txt`
   - Start command: `gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 1 --timeout 120 app:app`

## Notes

- LIAR is statement-level data; this predicts linguistic credibility style, not definitive fact verification.
- Keep `models/best_model.joblib` and `models/tfidf_vectorizer.joblib` available in deployment environment.
- SHAP/LIME can add latency for online inference; disable via request flags if needed.

## License

MIT. See [`LICENSE`](LICENSE).
