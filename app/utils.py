from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.explainability import build_explanation_payload
from app.preprocessing import preprocess_text


def predict_proba_compat(model, vectorizer, preprocessed_text: str) -> tuple[int, float]:
    X = vectorizer.transform([preprocessed_text])
    prediction = int(model.predict(X)[0])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        return prediction, float(probs[prediction])

    if hasattr(model, "decision_function"):
        decision = float(np.ravel(model.decision_function(X))[0])
        prob_real = float(1.0 / (1.0 + np.exp(-decision)))
        prob_fake = 1.0 - prob_real
        probs = [prob_fake, prob_real]
        return prediction, probs[prediction]

    return prediction, 0.5


def run_prediction(
    raw_text: str,
    model,
    vectorizer,
    include_shap: bool = True,
    include_lime: bool = True,
) -> dict[str, Any]:
    preprocessed_text = preprocess_text(raw_text)
    if not preprocessed_text:
        raise ValueError("Input text does not contain usable language tokens.")

    prediction, confidence = predict_proba_compat(model, vectorizer, preprocessed_text)
    label = "FAKE NEWS" if prediction == 0 else "REAL NEWS"

    explanation = build_explanation_payload(
        raw_text=raw_text,
        preprocessed_text=preprocessed_text,
        model=model,
        vectorizer=vectorizer,
        include_shap=include_shap,
        include_lime=include_lime,
    )

    return {
        "prediction": label,
        "prediction_id": prediction,
        "confidence": round(confidence * 100, 2),
        "preprocessed_text": preprocessed_text,
        "explanation": explanation,
    }


def fetch_article_text(url: str) -> str:
    try:
        from newspaper import Article  # type: ignore
    except Exception as exc:
        raise RuntimeError("newspaper3k is not installed. Install dependencies first.") from exc

    article = Article(url)
    article.download()
    article.parse()

    text = (article.text or "").strip()
    if not text:
        raise ValueError("Unable to extract article text from URL.")
    return text


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_jsonl(log_path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(log_path)
    ensure_parent_dir(path)
    record = dict(payload)
    record["timestamp"] = datetime.now(timezone.utc).isoformat()

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def read_jsonl(log_path: str | Path) -> list[dict[str, Any]]:
    path = Path(log_path)
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records

