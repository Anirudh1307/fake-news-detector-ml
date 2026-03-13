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

from app.apis.factcheck_client import search_claim_reviews
from app.apis.gnews_client import search_news
from app.explainability import build_explanation_payload
from app.preprocessing import preprocess_text

LOGGER = logging.getLogger(__name__)
WHITESPACE_PATTERN = re.compile(r"\s+")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
EXTRA_SYMBOL_PATTERN = re.compile(r"[^\w\s\.,;:!?\-\"'()/%&]")
TRUSTED_SOURCE_HOSTS = (
    "bbc.com",
    "reuters.com",
    "apnews.com",
    "nytimes.com",
    "theguardian.com",
)
TRUSTED_SOURCE_ALIASES = (
    "bbc",
    "reuters",
    "associated press",
    "ap news",
    "apnews",
    "new york times",
    "nytimes",
    "the guardian",
)
MISINFORMATION_PATTERNS = (
    ("secret cure", re.compile(r"\bsecret\W+cure\b", re.IGNORECASE)),
    ("miracle cure", re.compile(r"\bmiracle\W+cure\b", re.IGNORECASE)),
    ("government conspiracy", re.compile(r"\bgovernment\W+conspiracy\b", re.IGNORECASE)),
    ("shocking truth", re.compile(r"\bshocking\W+truth\b", re.IGNORECASE)),
    (
        "they don't want you to know",
        re.compile(r"\bthey\W+don\W*t\W+want\W+you\W+to\W+know\b", re.IGNORECASE),
    ),
    ("hidden technology", re.compile(r"\bhidden\W+technology\b", re.IGNORECASE)),
    ("leaked documents reveal", re.compile(r"\bleaked\W+documents\W+reveal\b", re.IGNORECASE)),
)
MIN_ARTICLE_CHARS = 200
MIN_ARTICLE_TOKENS = 30
KEYWORD_FAKE_BOOST_PER_MATCH = 0.03
MAX_KEYWORD_FAKE_BOOST = 0.15
UNCERTAIN_LOWER = 0.45
UNCERTAIN_UPPER = 0.55
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
ML_WEIGHT = 0.6
SOURCE_WEIGHT = 0.2
KEYWORD_WEIGHT = 0.1
FACTCHECK_WEIGHT = 0.1
NEUTRAL_SOURCE_SCORE = 0.5
NEUTRAL_FACTCHECK_SCORE = 0.5
DIRECT_TRUSTED_SOURCE_BONUS = 0.35
GNEWS_TRUSTED_MATCH_BONUS = 0.15


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


def _parse_source_host(source_url: str | None) -> str:
    if not source_url:
        return ""
    try:
        return (urlparse(source_url).hostname or "").lower()
    except Exception:
        return ""


def _is_trusted_source_name(source_name: str) -> bool:
    normalized = (source_name or "").strip().lower()
    if not normalized:
        return False
    return any(alias in normalized for alias in TRUSTED_SOURCE_ALIASES)


def _extract_keyword_query(raw_text: str, max_keywords: int = 8) -> tuple[str, set[str]]:
    processed = preprocess_text(raw_text, remove_stopwords=True, apply_lemmatization=False)
    tokens = [token for token in processed.split() if len(token) >= 3]

    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
        if len(deduped) >= max_keywords:
            break

    if not deduped:
        fallback_tokens = re.findall(r"[a-zA-Z]{4,}", raw_text.lower())
        for token in fallback_tokens:
            if token in seen:
                continue
            seen.add(token)
            deduped.append(token)
            if len(deduped) >= max_keywords:
                break

    return " ".join(deduped), set(deduped)


def _article_matches_keywords(article: dict[str, str], keyword_set: set[str]) -> bool:
    if not keyword_set:
        return False

    combined = f"{article.get('title', '')} {article.get('description', '')}".lower()
    article_tokens = set(re.findall(r"[a-zA-Z]{3,}", combined))
    overlap = len(article_tokens.intersection(keyword_set))
    minimum_overlap = 1 if len(keyword_set) <= 4 else 2
    return overlap >= minimum_overlap


def _extract_claim(raw_text: str) -> str:
    if not isinstance(raw_text, str):
        return ""

    sentences = [segment.strip() for segment in SENTENCE_SPLIT_PATTERN.split(raw_text) if segment.strip()]
    if not sentences:
        return raw_text.strip()

    longest = max(sentences, key=len)
    return longest or sentences[0]


def _factcheck_rating_to_real_score(rating: str) -> float:
    normalized = (rating or "").strip().lower()
    if "misleading" in normalized:
        return 0.2
    if "mostly true" in normalized:
        return 0.75
    if "mostly false" in normalized:
        return 0.15
    if "false" in normalized:
        return 0.0
    if "true" in normalized:
        return 1.0
    return NEUTRAL_FACTCHECK_SCORE


def _build_source_score(source_url: str | None, raw_text: str) -> tuple[float, bool, str, bool, int, str]:
    host = _parse_source_host(source_url)
    trusted_source_match = any(domain in host for domain in TRUSTED_SOURCE_HOSTS) if host else False

    source_score = NEUTRAL_SOURCE_SCORE + (DIRECT_TRUSTED_SOURCE_BONUS if trusted_source_match else 0.0)
    query, keyword_set = _extract_keyword_query(raw_text)
    gnews_articles: list[dict[str, str]] = []
    gnews_trusted_match = False

    if query:
        try:
            gnews_articles = search_news(query)
        except Exception:
            LOGGER.exception("gnews_enrichment_failed")
            gnews_articles = []

    for article in gnews_articles:
        source_name = article.get("source", "")
        if _is_trusted_source_name(source_name) and _article_matches_keywords(article, keyword_set):
            gnews_trusted_match = True
            break

    if gnews_trusted_match:
        source_score += GNEWS_TRUSTED_MATCH_BONUS

    return float(np.clip(source_score, 0.0, 1.0)), trusted_source_match, host, gnews_trusted_match, len(gnews_articles), query


def _build_keyword_score(raw_text: str) -> tuple[float, float, list[str]]:
    keyword_fake_boost, matched_keywords = _misinformation_fake_boost(raw_text)
    if MAX_KEYWORD_FAKE_BOOST <= 0:
        return 1.0, keyword_fake_boost, matched_keywords

    penalty_ratio = keyword_fake_boost / MAX_KEYWORD_FAKE_BOOST
    keyword_score = float(np.clip(1.0 - penalty_ratio, 0.0, 1.0))
    return keyword_score, keyword_fake_boost, matched_keywords


def _build_factcheck_score(raw_text: str) -> tuple[float, str, str, int]:
    claim = _extract_claim(raw_text)
    if not claim:
        return NEUTRAL_FACTCHECK_SCORE, "", "", 0

    try:
        reviews = search_claim_reviews(claim)
    except Exception:
        LOGGER.exception("factcheck_enrichment_failed")
        return NEUTRAL_FACTCHECK_SCORE, claim, "", 0

    if not reviews:
        return NEUTRAL_FACTCHECK_SCORE, claim, "", 0

    primary_rating = (reviews[0].get("rating") or "").strip()
    factcheck_score = _factcheck_rating_to_real_score(primary_rating)
    return factcheck_score, claim, primary_rating, len(reviews)


def _misinformation_fake_boost(raw_text: str) -> tuple[float, list[str]]:
    text = raw_text or ""
    matched = [label for label, pattern in MISINFORMATION_PATTERNS if pattern.search(text)]
    boost = min(MAX_KEYWORD_FAKE_BOOST, KEYWORD_FAKE_BOOST_PER_MATCH * len(matched))
    return float(boost), matched


def run_prediction(
    raw_text: str,
    model,
    vectorizer,
    include_shap: bool = True,
    include_lime: bool = True,
    source_url: str | None = None,
    preprocessed_text: str | None = None,
) -> dict[str, Any]:
    preprocessed_text = preprocessed_text if preprocessed_text is not None else preprocess_text(raw_text)
    if not preprocessed_text:
        raise ValueError("Input text does not contain usable language tokens.")

    ml_real_score = _ml_probability_real(model, vectorizer, preprocessed_text)
    (
        source_score,
        trusted_source,
        source_host,
        gnews_trusted_match,
        gnews_results_count,
        gnews_query,
    ) = _build_source_score(source_url, raw_text)
    keyword_score, keyword_fake_boost, matched_keywords = _build_keyword_score(raw_text)
    factcheck_score, extracted_claim, factcheck_rating, factcheck_results_count = _build_factcheck_score(raw_text)

    final_real_score = float(
        np.clip(
            (ML_WEIGHT * ml_real_score)
            + (SOURCE_WEIGHT * source_score)
            + (KEYWORD_WEIGHT * keyword_score)
            + (FACTCHECK_WEIGHT * factcheck_score),
            0.0,
            1.0,
        )
    )
    calibrated_probability = round(final_real_score, 2)

    prediction = int(calibrated_probability >= 0.5)
    confidence = calibrated_probability if prediction == 1 else 1.0 - calibrated_probability
    is_uncertain = UNCERTAIN_LOWER < calibrated_probability < UNCERTAIN_UPPER
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
        "source_host": source_host,
        "trusted_source_boost": round(max(0.0, source_score - NEUTRAL_SOURCE_SCORE), 4),
        "trusted_source_match": trusted_source,
        "source_score": round(source_score, 4),
        "gnews_query": gnews_query,
        "gnews_results_count": gnews_results_count,
        "gnews_trusted_match": gnews_trusted_match,
        "keyword_score": round(keyword_score, 4),
        "keyword_fake_boost": round(keyword_fake_boost, 4),
        "factcheck_score": round(factcheck_score, 4),
        "factcheck_rating": factcheck_rating,
        "factcheck_results_count": factcheck_results_count,
        "extracted_claim": extracted_claim,
        "final_real_score": round(final_real_score, 4),
        "calibrated_probability": calibrated_probability,
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
    cleaned = EXTRA_SYMBOL_PATTERN.sub(" ", cleaned)
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
        timeout=10,
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

    html = _download_html(url)
    doc = Document(html)
    return _clean_extracted_text(doc.summary())


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
        ("trafilatura", _extract_with_trafilatura),
        ("newspaper3k", _extract_with_newspaper),
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
