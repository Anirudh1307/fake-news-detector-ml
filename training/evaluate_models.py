from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _positive_probability(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        decision = np.ravel(model.decision_function(X))
        return 1.0 / (1.0 + np.exp(-decision))

    return None


def compute_classification_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def evaluate_models(
    models: dict[str, Any],
    X_test_vec,
    y_test,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    details = {}

    for name, model in models.items():
        y_prob = _positive_probability(model, X_test_vec)
        y_pred = model.predict(X_test_vec)

        metrics = compute_classification_metrics(y_test, y_pred)
        if y_prob is not None:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
        else:
            metrics["roc_auc"] = float("nan")

        rows.append({"model": name, **metrics})
        details[name] = {
            "y_pred": y_pred,
            "y_prob": y_prob,
            "classification_report": classification_report(y_test, y_pred, zero_division=0, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }

    comparison_df = pd.DataFrame(rows).sort_values(by="f1_score", ascending=False).reset_index(drop=True)
    return comparison_df, details


def _save_confusion_matrix_plot(cm, model_name: str, output_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _save_roc_plot(y_true, y_prob, model_name: str, output_path: Path) -> None:
    if y_prob is None:
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_score:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _save_accuracy_comparison(comparison_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.barplot(data=comparison_df, x="model", y="accuracy", hue="model", palette="viridis", legend=False)
    plt.ylim(0, 1)
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_evaluation_artifacts(
    comparison_df: pd.DataFrame,
    details: dict,
    y_test,
    reports_dir: str | Path,
) -> None:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    comparison_csv = reports_dir / "accuracy_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)
    _save_accuracy_comparison(comparison_df, reports_dir / "accuracy_comparison.png")

    for model_name, data in details.items():
        safe_name = model_name.lower().replace(" ", "_")
        cm_path = reports_dir / f"{safe_name}_confusion_matrix.png"
        _save_confusion_matrix_plot(data["confusion_matrix"], model_name, cm_path)

        roc_path = reports_dir / f"{safe_name}_roc_curve.png"
        _save_roc_plot(y_test, data["y_prob"], model_name, roc_path)

        report_path = reports_dir / f"{safe_name}_classification_report.csv"
        pd.DataFrame(data["classification_report"]).transpose().to_csv(report_path)
