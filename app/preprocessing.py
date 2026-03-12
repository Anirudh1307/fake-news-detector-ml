from __future__ import annotations

import re
import string
import unicodedata
from typing import Iterable, Sequence

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")
NUMBER_PATTERN = re.compile(r"\d+")
NON_ALPHA_PATTERN = re.compile(r"[^a-z\s]")
PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)
EXTRA_STOPWORDS = {
    "said",
    "say",
    "says",
    "reuters",
    "ap",
    "news",
    "report",
    "reports",
    "breaking",
}
DEFAULT_STOPWORDS = set(ENGLISH_STOP_WORDS).union(EXTRA_STOPWORDS)

_lemmatizer = None
_nltk_wordnet_available = False

try:
    import nltk
    from nltk.stem import WordNetLemmatizer

    _lemmatizer = WordNetLemmatizer()
    try:
        nltk.data.find("corpora/wordnet")
        _nltk_wordnet_available = True
    except LookupError:
        _nltk_wordnet_available = False
except Exception:
    _lemmatizer = None
    _nltk_wordnet_available = False


def _fallback_lemmatize(token: str) -> str:
    """Rule-based fallback when NLTK WordNet corpus is unavailable."""
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("sses") and len(token) > 5:
        return token[:-2]
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def lemmatize_token(token: str) -> str:
    if not token:
        return token
    if _lemmatizer is not None and _nltk_wordnet_available:
        try:
            return _lemmatizer.lemmatize(token)
        except Exception:
            return _fallback_lemmatize(token)
    return _fallback_lemmatize(token)


def preprocess_text(
    text: str,
    remove_stopwords: bool = True,
    apply_lemmatization: bool = True,
    stopwords: set[str] | None = None,
    extra_stopwords: set[str] | None = None,
) -> str:
    """Advanced NLP preprocessing for classification."""
    if not isinstance(text, str):
        return ""

    processed = unicodedata.normalize("NFKC", text)
    processed = processed.lower()
    processed = URL_PATTERN.sub(" ", processed)
    processed = processed.translate(PUNCT_TRANSLATION)
    processed = NUMBER_PATTERN.sub(" ", processed)
    processed = NON_ALPHA_PATTERN.sub(" ", processed)
    processed = WHITESPACE_PATTERN.sub(" ", processed).strip()

    if not processed:
        return ""

    tokens = processed.split()
    stopword_set = set(DEFAULT_STOPWORDS if stopwords is None else stopwords)
    if extra_stopwords:
        stopword_set.update(extra_stopwords)

    if remove_stopwords:
        tokens = [token for token in tokens if token not in stopword_set]

    if apply_lemmatization:
        tokens = [lemmatize_token(token) for token in tokens]

    return " ".join(tokens).strip()


def preprocess_corpus(
    texts: Iterable[str],
    remove_stopwords: bool = True,
    apply_lemmatization: bool = True,
) -> list[str]:
    return [
        preprocess_text(
            text,
            remove_stopwords=remove_stopwords,
            apply_lemmatization=apply_lemmatization,
        )
        for text in texts
    ]


def deduplicate_texts(texts: Sequence[str]) -> list[str]:
    unique_texts: list[str] = []
    seen: set[str] = set()
    for text in texts:
        if text not in seen:
            unique_texts.append(text)
            seen.add(text)
    return unique_texts


def deduplicate_dataframe(df, text_column: str):
    """Drop duplicate rows based on a preprocessed text column."""
    if text_column not in df.columns:
        raise ValueError(f"Missing text column '{text_column}'")
    return df.drop_duplicates(subset=[text_column]).reset_index(drop=True)
