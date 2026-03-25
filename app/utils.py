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

LOGGER = logging.getLogger(__name__)
WHITESPACE_PATTERN = re.compile(r"\s+")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
EXTRA_SYMBOL_PATTERN = re.compile(r"[^\w\s\.,;:!?\-\"'()/%&]")
TRUSTED_SOURCE_HOSTS = (
    "bbc.com",
    "reuters.com",
    "thehindu.com",
    "ndtv.com",
)
FAKE_SOURCE_HOSTS = (
    "beforeitsnews.com",
    "worldnewsdailyreport.com",
)
SOCIAL_SOURCE_HOSTS = (
    "youtube.com",
    "twitter.com",
    "x.com",
)
REAL_KEYWORDS = (
    "official",
    "report",
    "according",
    "data",
    "government",
)
FAKE_KEYWORDS = (
    "breaking",
    "shocking",
    "secret",
    "exposed",
    "cure",
    "miracle",
)
STRONG_FAKE_CLAIMS = (
    "teleport",
    "alien cure",
    "miracle cure",
    "time travel",
)
MIN_DIRECT_TEXT_CHARS = 20
MIN_URL_EXTRACTION_CHARS = 200
MIN_URL_WORDS = 50
LOW_QUALITY_PATTERNS = (
    "sign in",
    "subscribe",
    "advertisement",
    "ad blocker",
    "cookies",
    "newsletter",
    "login",
    "access benefits",
)
VAGUE_PATTERNS = (
    "many people",
    "some say",
    "it is believed",
    "reports suggest",
    "people are talking",
    "nothing confirmed",
    "details emerging",
)
UNCERTAIN_LOWER = 0.45
UNCERTAIN_UPPER = 0.55
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
ML_WEIGHT = 0.6
HYBRID_KEYWORD_WEIGHT = 0.2
HYBRID_SOURCE_WEIGHT = 0.2
NEUTRAL_FACTCHECK_SCORE = 0.5
TRUSTED_DOMAIN_SCORE_BOOST = 1.0
FAKE_DOMAIN_SCORE_PENALTY = -1.0
UNCERTAINTY_INDICATORS = (
    "may",
    "might",
    "claim",
    "reportedly",
    "unverified",
    "no proof",
)


def _ml_scores(model, vectorizer, preprocessed_text: str) -> tuple[float, float]:
    X = vectorizer.transform([preprocessed_text])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        prob_real = float(probs[1])
        confidence = float(np.max(probs))
        return prob_real, confidence

    if hasattr(model, "decision_function"):
        decision = float(np.ravel(model.decision_function(X))[0])
        prob_real = float(1.0 / (1.0 + np.exp(-decision)))
        return prob_real, max(prob_real, 1.0 - prob_real)

    prediction = int(model.predict(X)[0])
    prob_real = 1.0 if prediction == 1 else 0.0
    return prob_real, 1.0


def _parse_source_host(source_url: str | None) -> str:
    if not source_url:
        return ""
    try:
        return get_domain(source_url)
    except Exception:
        return ""


def is_domain_match(domain: str, sources: tuple[str, ...] | list[str]) -> bool:
    normalized = (domain or "").strip().lower()
    return any(normalized == site or normalized.endswith(f".{site}") for site in sources)


def get_domain(url: str) -> str:
    if not isinstance(url, str):
        return ""

    hostname = (urlparse(url).hostname or "").lower().strip()
    if hostname.startswith("www."):
        hostname = hostname[4:]
    return hostname


def classify_domain(url: str) -> dict[str, Any] | None:
    domain = get_domain(url)
    if not domain:
        return None

    if is_domain_match(domain, SOCIAL_SOURCE_HOSTS):
        return {
            "domain": domain,
            "classification": "SOCIAL",
            "prediction": "INSUFFICIENT_CONTEXT",
            "confidence": 0,
            "reason": "Social media content not suitable for analysis.",
        }

    if is_domain_match(domain, FAKE_SOURCE_HOSTS):
        return {
            "domain": domain,
            "classification": "FAKE_DOMAIN",
            "score_adjustment": FAKE_DOMAIN_SCORE_PENALTY,
            "reason": "Known low-credibility domain contributes a reliability penalty.",
        }

    if is_domain_match(domain, TRUSTED_SOURCE_HOSTS):
        return {
            "domain": domain,
            "classification": "TRUSTED_DOMAIN",
            "score_adjustment": TRUSTED_DOMAIN_SCORE_BOOST,
            "reason": "Trusted news/public-interest domain contributes a credibility boost.",
        }

    return None


def is_non_article_url(url: str) -> bool:
    if not isinstance(url, str):
        return True

    return url.count("/") <= 3


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


def _build_source_score(source_url: str | None) -> tuple[float, bool, bool, str, bool, int, str]:
    host = _parse_source_host(source_url)
    domain_assessment = classify_domain(source_url or "")
    trusted_source_match = bool(domain_assessment and domain_assessment.get("classification") == "TRUSTED_DOMAIN")
    fake_source_match = bool(domain_assessment and domain_assessment.get("classification") == "FAKE_DOMAIN")

    if trusted_source_match:
        source_score = TRUSTED_DOMAIN_SCORE_BOOST
    elif fake_source_match:
        source_score = FAKE_DOMAIN_SCORE_PENALTY
    else:
        source_score = 0.0

    return (
        float(np.clip(source_score, -1.0, 1.0)),
        trusted_source_match,
        fake_source_match,
        host,
        False,
        0,
        "",
    )


def _build_keyword_score(raw_text: str) -> tuple[float, float, list[str], list[str]]:
    text = str(raw_text or "").lower()
    real_matches = [word for word in REAL_KEYWORDS if re.search(rf"\b{re.escape(word)}\b", text)]
    fake_matches = [word for word in FAKE_KEYWORDS if re.search(rf"\b{re.escape(word)}\b", text)]
    real_score = len(real_matches) / len(REAL_KEYWORDS) if REAL_KEYWORDS else 0.0
    fake_score = len(fake_matches) / len(FAKE_KEYWORDS) if FAKE_KEYWORDS else 0.0
    keyword_score = float(np.clip(real_score - fake_score, -1.0, 1.0))
    return keyword_score, fake_score, real_matches, fake_matches


def get_top_words(vectorizer, text: str, top_n: int = 5) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []

    X = vectorizer.transform([text])
    if X.nnz == 0:
        return []

    feature_names = vectorizer.get_feature_names_out()
    scores = X.toarray()[0]
    top_indices = scores.argsort()[-top_n:][::-1]

    return [str(feature_names[i]) for i in top_indices if scores[i] > 0]


def _build_factcheck_score(raw_text: str) -> tuple[float, str, str, int]:
    claim = _extract_claim(raw_text)
    if not claim:
        return NEUTRAL_FACTCHECK_SCORE, "", "", 0

    try:
        from app.apis.factcheck_client import search_claim_reviews

        reviews = search_claim_reviews(claim)
    except Exception:
        LOGGER.exception("factcheck_enrichment_failed")
        return NEUTRAL_FACTCHECK_SCORE, claim, "", 0

    if not reviews:
        return NEUTRAL_FACTCHECK_SCORE, claim, "", 0

    primary_rating = (reviews[0].get("rating") or "").strip()
    factcheck_score = _factcheck_rating_to_real_score(primary_rating)
    return factcheck_score, claim, primary_rating, len(reviews)


def _contains_uncertainty(raw_text: str) -> bool:
    normalized = f" {str(raw_text or '').lower()} "
    return any(f" {indicator} " in normalized for indicator in UNCERTAINTY_INDICATORS)


def _contains_vague_language(raw_text: str) -> bool:
    normalized = str(raw_text or "").lower()
    return any(pattern in normalized for pattern in VAGUE_PATTERNS)


def _calibrate_confidence(prob: float, source_score: float) -> int:
    confidence = abs(float(prob) - 0.5) * 100
    if source_score != 0:
        confidence += 5
    return int(round(min(max(confidence, 40), 90)))


def run_prediction(
    raw_text: str,
    model,
    vectorizer,
    include_shap: bool = True,
    include_lime: bool = True,
    source_url: str | None = None,
    preprocessed_text: str | None = None,
) -> dict[str, Any]:
    from app.explainability import build_explanation_payload
    from app.preprocessing import preprocess_text

    preprocessed_text = preprocessed_text if preprocessed_text is not None else preprocess_text(raw_text)
    if not preprocessed_text:
        raise ValueError("Input text does not contain usable language tokens.")
    if len(str(raw_text or "").strip()) < MIN_DIRECT_TEXT_CHARS:
        return {
            "prediction": "INSUFFICIENT_CONTEXT",
            "prediction_id": -1,
            "confidence": 0,
            "preprocessed_text": preprocessed_text,
            "explanation": {"hybrid_signals": {}},
        }

    lowered_text = str(raw_text or "").lower()
    strong_fake_match = [phrase for phrase in STRONG_FAKE_CLAIMS if phrase in lowered_text]
    if strong_fake_match:
        explanation = build_explanation_payload(
            raw_text=raw_text,
            preprocessed_text=preprocessed_text,
            model=model,
            vectorizer=vectorizer,
            include_shap=include_shap,
            include_lime=include_lime,
        )
        explanation["hybrid_signals"] = {
            "ml_real_score": 0.0,
            "source_host": _parse_source_host(source_url),
            "trusted_source_boost": 0.0,
            "fake_source_penalty": 0.0,
            "trusted_source_match": False,
            "fake_source_match": False,
            "source_score": 0.0,
            "gnews_query": "",
            "gnews_results_count": 0,
            "gnews_trusted_match": False,
            "model_confidence": 85,
            "model_probability": 0.5,
            "keyword_score": 0.0,
            "keyword_fake_boost": 0.0,
            "matched_real_keywords": [],
            "factcheck_score": round(NEUTRAL_FACTCHECK_SCORE, 4),
            "factcheck_rating": "",
            "factcheck_results_count": 0,
            "extracted_claim": _extract_claim(raw_text),
            "final_score": -1.0,
            "matched_misinformation_keywords": [],
            "uncertainty_detected": False,
            "strong_fake_match": strong_fake_match,
        }
        return {
            "prediction": "FAKE",
            "prediction_id": 0,
            "confidence": 85,
            "preprocessed_text": preprocessed_text,
            "explanation": explanation,
        }

    ml_real_score, _ = _ml_scores(model, vectorizer, preprocessed_text)
    (
        source_score,
        trusted_source,
        fake_source,
        source_host,
        gnews_trusted_match,
        gnews_results_count,
        gnews_query,
    ) = _build_source_score(source_url)
    keyword_score, keyword_fake_boost, matched_real_keywords, matched_fake_keywords = _build_keyword_score(raw_text)
    factcheck_score, extracted_claim, factcheck_rating, factcheck_results_count = _build_factcheck_score(raw_text)

    prob = float(ml_real_score)
    model_score = (prob - 0.5) * 2.0
    final_score = (
        (ML_WEIGHT * model_score)
        + (HYBRID_KEYWORD_WEIGHT * keyword_score)
        + (HYBRID_SOURCE_WEIGHT * source_score)
    )
    confidence = _calibrate_confidence(prob, source_score)
    vague_detected = _contains_vague_language(raw_text)
    uncertainty_detected = _contains_uncertainty(raw_text) or (45 <= confidence <= 60)

    if vague_detected:
        label = "UNCERTAIN"
        prediction_id = -1
        confidence = 45
    elif confidence <= 45 and keyword_score == 0 and not matched_fake_keywords:
        label = "REAL"
        prediction_id = 1
        confidence = 40
    elif confidence < 45:
        if keyword_score == 0:
            label = "REAL"
            prediction_id = 1
            confidence = 40
        else:
            label = "UNCERTAIN"
            prediction_id = -1
    elif uncertainty_detected:
        label = "UNCERTAIN"
        prediction_id = -1
        confidence = max(45, min(confidence, 60))
    elif final_score > 0:
        label = "REAL"
        prediction_id = 1
    else:
        label = "FAKE"
        prediction_id = 0

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
        "trusted_source_boost": round(source_score if trusted_source and source_score > 0 else 0.0, 4),
        "fake_source_penalty": round(abs(source_score) if fake_source and source_score < 0 else 0.0, 4),
        "trusted_source_match": trusted_source,
        "fake_source_match": fake_source,
        "source_score": round(source_score, 4),
        "gnews_query": gnews_query,
        "gnews_results_count": gnews_results_count,
        "gnews_trusted_match": gnews_trusted_match,
        "model_confidence": confidence,
        "model_probability": round(prob, 4),
        "keyword_score": round(keyword_score, 4),
        "keyword_fake_boost": round(keyword_fake_boost, 4),
        "matched_real_keywords": matched_real_keywords,
        "factcheck_score": round(factcheck_score, 4),
        "factcheck_rating": factcheck_rating,
        "factcheck_results_count": factcheck_results_count,
        "extracted_claim": extracted_claim,
        "final_score": round(final_score, 4),
        "matched_misinformation_keywords": matched_fake_keywords,
        "uncertainty_detected": uncertainty_detected,
        "vague_detected": vague_detected,
        "strong_fake_match": strong_fake_match,
    }

    return {
        "prediction": label.upper(),
        "prediction_id": prediction_id,
        "confidence": confidence,
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


def assess_extracted_article_text(text: str) -> dict[str, Any]:
    cleaned = _clean_extracted_text(text)
    lowered = cleaned.lower()
    word_count = len(cleaned.split())
    bad_pattern_count = sum(1 for pattern in LOW_QUALITY_PATTERNS if pattern in lowered)
    return {
        "cleaned_text": cleaned,
        "char_count": len(cleaned),
        "word_count": word_count,
        "bad_pattern_count": bad_pattern_count,
        "is_short_for_fallback": len(cleaned) < MIN_URL_EXTRACTION_CHARS,
        "is_low_quality": word_count < MIN_URL_WORDS or bad_pattern_count >= 2,
    }


def _download_html(url: str) -> str:
    import requests  # type: ignore

    response = requests.get(
        url,
        timeout=10,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "en-US,en;q=0.9",
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


def _extract_with_trafilatura(url: str, html: str | None = None) -> str:
    import trafilatura  # type: ignore

    downloaded = html if isinstance(html, str) and html else trafilatura.fetch_url(url)
    if not downloaded:
        return ""
    extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    return _clean_extracted_text(extracted or "")


def _extract_with_readability(url: str) -> str:
    from readability import Document  # type: ignore

    html = _download_html(url)
    doc = Document(html)
    return _clean_extracted_text(doc.summary())


def _extract_with_bs4(url: str, html: str | None = None) -> str:
    from bs4 import BeautifulSoup  # type: ignore

    html = html if isinstance(html, str) and html else _download_html(url)
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p") if len(p.get_text(" ", strip=True)) > 30]
    text = " ".join(paragraphs)
    return _clean_extracted_text(text)


def fetch_article_text(url: str) -> str | dict[str, Any]:
    html = ""
    try:
        html = _download_html(url)
    except Exception:
        LOGGER.exception("html_download_failed url=%s", url)

    extraction_stages = [
        ("trafilatura", lambda target_url: _extract_with_trafilatura(target_url, html=html)),
        ("newspaper3k", _extract_with_newspaper),
        ("bs4", lambda target_url: _extract_with_bs4(target_url, html=html)),
    ]
    saw_low_quality_text = False

    for stage_name, extractor in extraction_stages:
        try:
            extracted = extractor(url)
            quality = assess_extracted_article_text(extracted)
            cleaned_text = quality["cleaned_text"]

            if cleaned_text and not quality["is_short_for_fallback"] and not quality["is_low_quality"]:
                LOGGER.info("extractor=%s url=%s", stage_name, url)
                return cleaned_text

            if cleaned_text:
                saw_low_quality_text = True

            LOGGER.warning(
                "extractor=%s weak_text url=%s chars=%s words=%s bad_patterns=%s",
                stage_name,
                url,
                quality["char_count"],
                quality["word_count"],
                quality["bad_pattern_count"],
            )
        except Exception:
            LOGGER.exception("extractor=%s failed url=%s", stage_name, url)

    if saw_low_quality_text:
        return {
            "prediction": "UNCERTAIN",
            "confidence": 45,
            "error": "Low quality extracted content",
        }

    return {
        "prediction": "UNCERTAIN",
        "confidence": 40,
        "error": "Unable to extract article",
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
