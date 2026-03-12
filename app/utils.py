from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any
import re
from urllib.parse import urlparse

import numpy as np

from app.explainability import build_explanation_payload
from app.preprocessing import preprocess_text

LOGGER = logging.getLogger(__name__)
WHITESPACE_PATTERN = re.compile(r"\s+")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
TRUSTED_SOURCE_HOSTS = (
    "bbc.com",
    "reuters.com",
    "apnews.com",
    "nytimes.com",
    "theguardian.com",
)
MISINFORMATION_KEYWORDS = (
    "secret cure",
    "shocking truth",
    "government conspiracy",
    "they don't want you to know",
)
MIN_ARTICLE_CHARS = 200
MIN_ARTICLE_TOKENS = 30


def _ml_probability_real(model, vectorizer, preprocessed_text: str) -> float:
    X = vectorizer.transform([preprocessed_text])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        return float(probs[1])

    if hasattr(model, "decision_function"):
        decision = float(np.ravel(model.decision_function(X))[0])
        return float(1.0 / (1.0 + np.exp(-decision)))

    prediction = int(model.predict(X)[0])
    return 1.0 if prediction == 1 else 0.0


def _source_credibility(source_url: str | None) -> float:
    if not source_url:
        return 0.5
    try:
        host = (urlparse(source_url).hostname or "").lower()
    except Exception:
        return 0.5
    return 1.0 if any(domain in host for domain in TRUSTED_SOURCE_HOSTS) else 0.5


def _keyword_credibility(raw_text: str) -> tuple[float, list[str]]:
    text = raw_text.lower()
    matched = [keyword for keyword in MISINFORMATION_KEYWORDS if keyword in text]
    score = max(0.0, 1.0 - (0.2 * len(matched)))
    return float(score), matched


def run_prediction(
    raw_text: str,
    model,
    vectorizer,
    include_shap: bool = True,
    include_lime: bool = True,
    source_url: str | None = None,
) -> dict[str, Any]:
    preprocessed_text = preprocess_text(raw_text)
    if not preprocessed_text:
        raise ValueError("Input text does not contain usable language tokens.")

    ml_real_score = _ml_probability_real(model, vectorizer, preprocessed_text)
    source_score = _source_credibility(source_url)
    keyword_score, matched_keywords = _keyword_credibility(raw_text)
    final_real_score = float(np.clip((0.8 * ml_real_score) + (0.1 * source_score) + (0.1 * keyword_score), 0.0, 1.0))

    prediction = int(final_real_score >= 0.5)
    confidence = final_real_score if prediction == 1 else 1.0 - final_real_score
    is_uncertain = 0.48 < confidence < 0.52
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
    explanation["hybrid_signals"] = {
        "ml_real_score": round(ml_real_score, 4),
        "source_credibility": round(source_score, 4),
        "keyword_credibility": round(keyword_score, 4),
        "final_real_score": round(final_real_score, 4),
        "matched_misinformation_keywords": matched_keywords,
    }

    return {
        "prediction": label,
        "prediction_id": prediction_id,
        "confidence": round(confidence * 100, 2),
        "preprocessed_text": preprocessed_text,
        "explanation": explanation,
    }


def _clean_extracted_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = unescape(text).replace("\xa0", " ")
    cleaned = HTML_TAG_PATTERN.sub(" ", cleaned)
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned


def _is_sufficient_article_text(text: str) -> bool:
    cleaned = _clean_extracted_text(text)
    if len(cleaned) < MIN_ARTICLE_CHARS:
        return False
    if len(cleaned.split()) < MIN_ARTICLE_TOKENS:
        return False
    return True


def _download_html(url: str) -> str:
    import requests  # type: ignore

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
    return response.text


def _extract_with_newspaper(url: str) -> str:
    from newspaper import Article  # type: ignore

    article = Article(url)
    article.download()
    article.parse()
    return _clean_extracted_text(article.text or "")


def _extract_with_trafilatura(url: str) -> str:
    import trafilatura  # type: ignore

    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return ""
    extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    return _clean_extracted_text(extracted or "")


def _extract_with_readability(url: str) -> str:
    from readability import Document  # type: ignore
    from bs4 import BeautifulSoup  # type: ignore

    html = _download_html(url)
    doc = Document(html)
    summary_html = doc.summary(html_partial=True)
    soup = BeautifulSoup(summary_html, "html.parser")
    text = soup.get_text(" ", strip=True)
    return _clean_extracted_text(text)


def _extract_with_bs4(url: str) -> str:
    from bs4 import BeautifulSoup  # type: ignore

    html = _download_html(url)
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = " ".join(paragraphs)
    return _clean_extracted_text(text)


def fetch_article_text(url: str) -> str | dict[str, Any]:
    extraction_stages = [
        ("newspaper3k", _extract_with_newspaper),
        ("trafilatura", _extract_with_trafilatura),
        ("readability", _extract_with_readability),
        ("bs4", _extract_with_bs4),
    ]

    for stage_name, extractor in extraction_stages:
        try:
            extracted = extractor(url)
            if _is_sufficient_article_text(extracted):
                LOGGER.info("extractor=%s url=%s", stage_name, url)
                return _clean_extracted_text(extracted)
            LOGGER.warning(
                "extractor=%s insufficient_text url=%s chars=%s tokens=%s",
                stage_name,
                url,
                len(extracted),
                len(extracted.split()),
            )
        except Exception:
            LOGGER.exception("extractor=%s failed url=%s", stage_name, url)

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
