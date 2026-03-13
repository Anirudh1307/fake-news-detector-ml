from __future__ import annotations

import logging
import os
from typing import Any

import requests

LOGGER = logging.getLogger(__name__)
GNEWS_SEARCH_ENDPOINT = "https://gnews.io/api/v4/search"
REQUEST_TIMEOUT_SECONDS = 8


def search_news(query: str) -> list[dict[str, str]]:
    if not isinstance(query, str) or not query.strip():
        return []

    api_key = os.getenv("GNEWS_API_KEY", "").strip()
    if not api_key:
        return []

    params = {
        "q": query.strip(),
        "lang": "en",
        "max": 5,
        "apikey": api_key,
    }

    try:
        response = requests.get(GNEWS_SEARCH_ENDPOINT, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
    except Exception:
        LOGGER.exception("gnews_lookup_failed")
        return []

    normalized: list[dict[str, str]] = []
    for article in payload.get("articles", []) or []:
        source_name = ""
        source_payload = article.get("source")
        if isinstance(source_payload, dict):
            source_name = str(source_payload.get("name", "")).strip()

        normalized.append(
            {
                "title": str(article.get("title", "")).strip(),
                "description": str(article.get("description", "")).strip(),
                "source": source_name,
                "url": str(article.get("url", "")).strip(),
            }
        )

    return normalized

