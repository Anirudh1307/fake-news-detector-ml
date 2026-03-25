from pathlib import Path

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from app.preprocessing import preprocess_corpus
from app.routes import create_app


def _long_article_text() -> str:
    return (
        "Official government data according to a detailed report from the finance ministry shows steady "
        "employment growth across several states. The report explains that public records, audited figures, "
        "and district level data were reviewed over multiple months. Officials said the findings match "
        "independent surveys, and the document includes dates, named sources, and methodology for each "
        "section of the analysis so readers can verify the context without relying on rumors or anonymous posts."
    )


def _build_test_model_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    texts = [
        "This claim is entirely fabricated and false",
        "Official records confirm the statement is accurate",
        "Fake conspiracy stories spread quickly online",
        "Verified report from credible public source",
    ]
    labels = [0, 1, 0, 1]

    processed = preprocess_corpus(texts)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(processed)
    model = LogisticRegression(solver="liblinear", random_state=42)
    model.fit(X, labels)

    model_path = tmp_path / "best_model.joblib"
    vectorizer_path = tmp_path / "tfidf_vectorizer.joblib"
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    return model_path, vectorizer_path


@pytest.fixture()
def test_app(tmp_path):
    model_path, vectorizer_path = _build_test_model_artifacts(tmp_path)
    app = create_app(
        {
            "TESTING": True,
            "MODEL_PATH": str(model_path),
            "VECTORIZER_PATH": str(vectorizer_path),
            "ANALYTICS_LOG_PATH": str(tmp_path / "prediction_logs.jsonl"),
            "ARTICLE_FETCHER": lambda _: _long_article_text(),
        }
    )
    return app


@pytest.fixture()
def client(test_app):
    with test_app.test_client() as test_client:
        yield test_client
