from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.svm import LinearSVC

from app.model_wrappers import CalibratedHybridModel
from training.dataset_loader import load_binary_dataset
from training.evaluate_models import evaluate_models, save_evaluation_artifacts


def parse_args():
    parser = argparse.ArgumentParser(description="Train and compare multiple fake news classifiers.")
    parser.add_argument("--dataset", default="isot", choices=["isot"], help="Dataset type")
    parser.add_argument("--fake-path", default="data/Fake.csv", help="Path to Fake.csv")
    parser.add_argument("--true-path", default="data/True.csv", help="Path to True.csv")
    parser.add_argument("--models-dir", default="models", help="Output directory for model artifacts")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--min-tokens", type=int, default=50, help="Drop rows with fewer article tokens")
    return parser.parse_args()


def build_models(random_state: int) -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(
            solver="liblinear",
            random_state=random_state,
            class_weight="balanced",
            max_iter=1000,
        ),
        "Linear SVM": LinearSVC(
            random_state=random_state,
            class_weight="balanced",
            max_iter=10000,
        ),
        "Passive Aggressive": PassiveAggressiveClassifier(
            random_state=random_state,
            max_iter=1000,
            tol=1e-3,
            class_weight="balanced",
        ),
    }


def select_best_model(comparison_df):
    ranking = comparison_df.sort_values(by=["f1_score", "accuracy"], ascending=False)
    return ranking.iloc[0]["model"]


def main():
    args = parse_args()
    if args.test_size <= 0 or args.test_size >= 1:
        raise ValueError("test-size must be > 0 and < 1.")

    models_dir = Path(args.models_dir)
    reports_dir = models_dir / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_binary_dataset(
        args.fake_path,
        args.true_path,
        min_tokens=args.min_tokens,
        dataset=args.dataset,
    )
    print(f"Loaded dataset with {len(dataset.frame)} rows")
    print(f"Dataset class distribution (fake=0, real=1): {dataset.y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X,
        dataset.y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=dataset.y,
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=5,
        max_features=40000,
        sublinear_tf=True,
        dtype=np.float32,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"Vectorizer vocabulary size: {len(vectorizer.vocabulary_)}")
    print(
        "TF-IDF params: stop_words=english, ngram=(1,2), max_df=0.9, "
        "min_df=5, max_features=40000, sublinear_tf=True"
    )

    models = build_models(args.random_state)
    trained_models: dict[str, object] = {}
    model_training_accuracy: dict[str, float] = {}
    model_best_params: dict[str, dict[str, float | int | str]] = {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.random_state)

    for name, base_model in models.items():
        print(f"Training {name}...")
        if name == "Logistic Regression":
            grid = GridSearchCV(
                estimator=base_model,
                param_grid={"C": [0.1, 1, 5, 10], "penalty": ["l1", "l2"]},
                scoring="f1",
                cv=cv,
                n_jobs=1,
                refit=True,
            )
            grid.fit(X_train_vec, y_train)
            model = grid.best_estimator_
            model_best_params[name] = grid.best_params_
        elif name == "Linear SVM":
            grid = GridSearchCV(
                estimator=base_model,
                param_grid={"C": [0.5, 1, 2], "loss": ["hinge", "squared_hinge"]},
                scoring="f1",
                cv=cv,
                n_jobs=1,
                refit=True,
            )
            grid.fit(X_train_vec, y_train)
            model = grid.best_estimator_
            model_best_params[name] = grid.best_params_
        elif name == "Passive Aggressive":
            grid = GridSearchCV(
                estimator=base_model,
                param_grid={"C": [0.1, 0.5, 1.0], "loss": ["hinge", "squared_hinge"]},
                scoring="f1",
                cv=cv,
                n_jobs=1,
                refit=True,
            )
            grid.fit(X_train_vec, y_train)
            model = grid.best_estimator_
            model_best_params[name] = grid.best_params_
        else:
            model = base_model
            model.fit(X_train_vec, y_train)
            model_best_params[name] = {}

        trained_models[name] = model
        model_training_accuracy[name] = float(model.score(X_train_vec, y_train))

    comparison_df, details = evaluate_models(trained_models, X_test_vec, y_test)
    print("\nModel comparison:")
    print(comparison_df.to_string(index=False))

    best_model_name = select_best_model(comparison_df)
    best_model = trained_models[best_model_name]
    print(f"\nBest model selected: {best_model_name}")
    print("Calibrating best model probabilities with CalibratedClassifierCV...")
    calibrator = CalibratedClassifierCV(
        estimator=best_model,
        method="sigmoid",
        cv=3,
    )
    calibrator.fit(X_train_vec, y_train)
    serving_model = CalibratedHybridModel(base_model=best_model, calibrator=calibrator)
    model_training_accuracy[best_model_name] = float(
        accuracy_score(y_train, serving_model.predict(X_train_vec))
    )

    save_evaluation_artifacts(comparison_df, details, y_test, reports_dir)
    print(f"Saved evaluation artifacts to: {reports_dir}")

    best_model_path = models_dir / "best_model.joblib"
    vectorizer_path = models_dir / "tfidf_vectorizer.joblib"
    metadata_path = models_dir / "metadata.joblib"

    joblib.dump(serving_model, best_model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(
        {
            "best_model_name": best_model_name,
            "model_scores": comparison_df.to_dict(orient="records"),
            "vectorizer_params": vectorizer.get_params(),
            "dataset_size": len(dataset.frame),
            "dataset_class_distribution": {str(k): int(v) for k, v in dataset.y.value_counts().to_dict().items()},
            "split_sizes": {
                "train": int(len(X_train)),
                "test": int(len(X_test)),
            },
            "dataset": args.dataset,
            "dataset_label_mapping": {"fake": 0, "real": 1},
            "preprocessing": {
                "min_tokens": int(args.min_tokens),
                "deduplicate": True,
                "lowercase": True,
                "remove_urls": True,
                "remove_numbers": True,
                "remove_punctuation": True,
                "normalize_whitespace": True,
            },
            "model_type": best_model_name,
            "dataset_used": args.dataset,
            "training_accuracy": model_training_accuracy[best_model_name],
            "training_date": datetime.now(timezone.utc).isoformat(),
            "best_params": model_best_params.get(best_model_name, {}),
            "probability_calibration": {
                "enabled": True,
                "method": "sigmoid",
                "cv": 3,
            },
        },
        metadata_path,
    )
    print(f"Saved best model to: {best_model_path}")
    print(f"Saved vectorizer to: {vectorizer_path}")
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
