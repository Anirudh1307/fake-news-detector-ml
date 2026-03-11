from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.preprocessing import deduplicate_dataframe, preprocess_corpus

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


def load_binary_dataset(fake_path: str | Path, true_path: str | Path) -> DatasetBundle:
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
    dataset["raw_text"] = (dataset["title"] + " " + dataset["text"]).str.strip()

    dataset["processed_text"] = preprocess_corpus(dataset["raw_text"].tolist())
    dataset = dataset[dataset["processed_text"] != ""].copy()
    dataset = deduplicate_dataframe(dataset, text_column="processed_text")

    return DatasetBundle(
        X=dataset["processed_text"],
        y=dataset["label"],
        frame=dataset,
    )

