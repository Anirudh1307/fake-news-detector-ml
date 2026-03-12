from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import re

import numpy as np

from app.explainability import build_explanation_payload
from app.preprocessing import preprocess_text

LOGGER = logging.getLogger(__name__)
WHITESPACE_PATTERN = re.compile(r"\s+")


def predict_proba_compat(model, vectorizer, preprocessed_text: str) -> tuple[int, float]:
    X = vectorizer.transform([preprocessed_text])
    threshold = float(getattr(model, "_decision_threshold", 0.5))

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        prob_real = float(probs[1])
        prediction = int(prob_real >= threshold)
        confidence = prob_real if prediction == 1 else 1.0 - prob_real
        return prediction, confidence

    if hasattr(model, "decision_function"):
        decision = float(np.ravel(model.decision_function(X))[0])
        prob_real = float(1.0 / (1.0 + np.exp(-decision)))
        prediction = int(prob_real >= threshold)
        confidence = prob_real if prediction == 1 else 1.0 - prob_real
        return prediction, confidence

    prediction = int(model.predict(X)[0])
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
    is_uncertain = 0.45 < confidence < 0.55
    label = "UNCERTAIN" if is_uncertain else ("FAKE NEWS" if prediction == 0 else "REAL NEWS")
    prediction_id = -1 if is_uncertain else prediction

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
        "prediction_id": prediction_id,
        "confidence": round(confidence * 100, 2),
        "preprocessed_text": preprocessed_text,
        "explanation": explanation,
    }


def _extract_article_with_requests(url: str) -> str:
    import requests  # type: ignore
    from bs4 import BeautifulSoup  # type: ignore

    response = requests.get(
        url,
        timeout=12,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        },
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = WHITESPACE_PATTERN.sub(" ", " ".join(paragraphs)).strip()
    if len(text.split()) < 50:
        raise ValueError("Extracted content is too short.")
    return text


def fetch_article_text(url: str) -> str | dict[str, Any]:
    article_cls = None
    try:
        from newspaper import Article  # type: ignore
        article_cls = Article
    except ImportError as exc:
        LOGGER.warning("newspaper3k unavailable, falling back to requests+bs4 extraction: %s", exc)

    if article_cls is not None:
        try:
            article = article_cls(url)
            article.download()
            article.parse()
            text = (article.text or "").strip()
            if len(text.split()) >= 50:
                return text
        except Exception:
            LOGGER.exception("newspaper3k extraction failed. url=%s", url)

    try:
        return _extract_article_with_requests(url)
    except Exception:
        LOGGER.exception("requests+bs4 extraction failed. url=%s", url)
        return {
            "error": "Unable to extract article text from this URL. The site may block scraping or require JavaScript.",
            "status_code": 400,
        }


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
