from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import string

import pandas as pd

REQUIRED_COLUMNS = {"title", "text"}
WHITESPACE_PATTERN = re.compile(r"\s+")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)


@dataclass
class DatasetBundle:
    X: pd.Series
    y: pd.Series
    frame: pd.DataFrame


def _validate_columns(df: pd.DataFrame, path: Path) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{path} missing required columns: {missing_text}")


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = text.translate(PUNCT_TRANSLATION)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def load_binary_dataset(
    fake_path: str | Path,
    true_path: str | Path,
    min_tokens: int = 50,
    dataset: str = "isot",
) -> DatasetBundle:
    if dataset.lower() != "isot":
        raise ValueError("Unsupported dataset. Use --dataset isot.")

    fake_path = Path(fake_path)
    true_path = Path(true_path)

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    _validate_columns(fake_df, fake_path)
    _validate_columns(true_df, true_path)

    fake_df = fake_df[["title", "text"]].copy()
    true_df = true_df[["title", "text"]].copy()
    fake_df["label"] = 1
    true_df["label"] = 0

    dataset = pd.concat([fake_df, true_df], ignore_index=True)
    dataset["title"] = dataset["title"].fillna("").astype(str)
    dataset["text"] = dataset["text"].fillna("").astype(str)
    dataset["article_text"] = (dataset["title"] + " " + dataset["text"]).str.strip().map(_normalize_text)
    dataset = dataset[dataset["article_text"] != ""].copy()
    dataset["token_count"] = dataset["article_text"].str.split().str.len()
    dataset = dataset[dataset["token_count"] >= max(1, int(min_tokens))].copy()
    dataset = dataset.drop_duplicates(subset=["article_text"]).reset_index(drop=True)

    return DatasetBundle(
        X=dataset["article_text"],
        y=dataset["label"],
        frame=dataset,
    )
