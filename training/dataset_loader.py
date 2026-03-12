from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.preprocessing import preprocess_text

REQUIRED_COLUMNS = {"title", "text"}


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
    fake_df["label"] = 0
    true_df["label"] = 1

    dataset = pd.concat([fake_df, true_df], ignore_index=True)
    dataset["title"] = dataset["title"].fillna("").astype(str)
    dataset["text"] = dataset["text"].fillna("").astype(str)
    dataset["text"] = dataset["text"].str.strip()
    dataset = dataset[dataset["text"] != ""].copy()

    dataset["content"] = (dataset["title"] + " " + dataset["text"]).str.strip()
    dataset["content_clean"] = dataset["content"].map(
        lambda value: preprocess_text(
            value,
            remove_stopwords=False,
            apply_lemmatization=False,
        )
    )
    dataset = dataset[dataset["content_clean"] != ""].copy()
    dataset["token_count"] = dataset["content_clean"].str.split().str.len()
    dataset = dataset[dataset["token_count"] >= max(1, int(min_tokens))].copy()
    dataset["content"] = dataset["content"].map(preprocess_text)
    dataset = dataset[dataset["content"] != ""].copy()
    dataset = dataset.drop_duplicates(subset=["content"]).reset_index(drop=True)

    return DatasetBundle(
        X=dataset["content"],
        y=dataset["label"],
        frame=dataset,
    )
