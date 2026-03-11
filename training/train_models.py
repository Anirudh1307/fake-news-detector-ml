from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from training.dataset_loader import load_binary_dataset
from training.evaluate_models import evaluate_models, save_evaluation_artifacts


def parse_args():
    parser = argparse.ArgumentParser(description="Train and compare multiple fake news classifiers.")
    parser.add_argument("--fake-path", default="data/Fake.csv", help="Path to Fake.csv")
    parser.add_argument("--true-path", default="data/True.csv", help="Path to True.csv")
    parser.add_argument("--models-dir", default="models", help="Output directory for model artifacts")
    parser.add_argument("--test-size", type=float, default=0.25, help="Test split ratio")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def build_models(random_state: int) -> dict:
    return {
        "Logistic Regression": LogisticRegression(solver="liblinear", random_state=random_state),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=random_state,
            n_jobs=1,
        ),
    }


def select_best_model(comparison_df):
    ranking = comparison_df.sort_values(by=["f1_score", "accuracy", "precision", "recall"], ascending=False)
    return ranking.iloc[0]["model"]


def main():
    args = parse_args()
    models_dir = Path(args.models_dir)
    reports_dir = models_dir / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_binary_dataset(args.fake_path, args.true_path)
    print(f"Loaded dataset with {len(dataset.frame)} rows")
    print(f"Class distribution: {dataset.y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X,
        dataset.y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=dataset.y,
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.8, ngram_range=(1, 2), min_df=2)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"Vectorizer vocabulary size: {len(vectorizer.vocabulary_)}")

    models = build_models(args.random_state)
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_vec, y_train)

    comparison_df, details = evaluate_models(models, X_test_vec, y_test)
    print("\nModel comparison:")
    print(comparison_df.to_string(index=False))

    best_model_name = select_best_model(comparison_df)
    best_model = models[best_model_name]
    print(f"\nBest model selected: {best_model_name}")

    save_evaluation_artifacts(comparison_df, details, y_test, reports_dir)
    print(f"Saved evaluation artifacts to: {reports_dir}")

    best_model_path = models_dir / "best_model.joblib"
    vectorizer_path = models_dir / "tfidf_vectorizer.joblib"
    metadata_path = models_dir / "metadata.joblib"

    joblib.dump(best_model, best_model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(
        {
            "best_model_name": best_model_name,
            "model_scores": comparison_df.to_dict(orient="records"),
            "vectorizer_params": vectorizer.get_params(),
            "dataset_size": len(dataset.frame),
        },
        metadata_path,
    )
    print(f"Saved best model to: {best_model_path}")
    print(f"Saved vectorizer to: {vectorizer_path}")
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
