from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Optional DistilBERT training for fake news detection.")
    parser.add_argument("--fake-path", default="data/Fake.csv")
    parser.add_argument("--true-path", default="data/True.csv")
    parser.add_argument("--output-dir", default="models/distilbert")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main():
    try:
        from datasets import Dataset
        from sklearn.model_selection import train_test_split
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        raise RuntimeError(
            "DistilBERT dependencies are missing. Install transformers, datasets, and torch."
        ) from exc

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fake = pd.read_csv(args.fake_path)
    true = pd.read_csv(args.true_path)
    fake["label"] = 0
    true["label"] = 1
    df = pd.concat([fake, true], ignore_index=True)
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["content"] = (df["title"] + " " + df["text"]).str.strip()
    df = df[df["content"] != ""].copy()

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    train_dataset = Dataset.from_pandas(train_df[["content", "label"]].reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df[["content", "label"]].reset_index(drop=True))

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def tokenize(batch):
        return tokenizer(batch["content"], truncation=True, max_length=256)

    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "runs"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved DistilBERT model to {output_dir}")


if __name__ == "__main__":
    main()
