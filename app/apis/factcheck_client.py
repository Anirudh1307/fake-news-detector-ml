from __future__ import annotations

import logging
import os
from typing import Any

import requests

LOGGER = logging.getLogger(__name__)
FACTCHECK_ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
REQUEST_TIMEOUT_SECONDS = 8


def search_claim_reviews(query: str) -> list[dict[str, str]]:
    if not isinstance(query, str) or not query.strip():
        return []

    api_key = os.getenv("GOOGLE_FACTCHECK_API_KEY", "").strip()
    if not api_key:
        return []

    params = {
        "query": query.strip(),
        "key": api_key,
    }

    try:
        response = requests.get(FACTCHECK_ENDPOINT, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
    except Exception:
        LOGGER.exception("factcheck_lookup_failed")
        return []

    normalized: list[dict[str, str]] = []
    for claim in payload.get("claims", []) or []:
        claim_text = str(claim.get("text", "")).strip()
        for review in claim.get("claimReview", []) or []:
            publisher_payload = review.get("publisher")
            publisher_name = ""
            if isinstance(publisher_payload, dict):
                publisher_name = str(publisher_payload.get("name", "")).strip()

            normalized.append(
                {
                    "claim": claim_text,
                    "rating": str(review.get("textualRating", "")).strip(),
                    "publisher": publisher_name,
                    "url": str(review.get("url", "")).strip(),
                }
            )

    return normalized

