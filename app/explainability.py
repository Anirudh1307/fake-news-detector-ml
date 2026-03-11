from __future__ import annotations

from typing import Callable

import numpy as np

from app.preprocessing import preprocess_text


def _predict_proba_compat(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)

    if hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        decision = np.atleast_2d(decision).reshape(-1, 1)
        probs_real = 1.0 / (1.0 + np.exp(-decision))
        probs_fake = 1.0 - probs_real
        return np.hstack([probs_fake, probs_real])

    raise ValueError("Model does not support probability or decision scores.")


def get_global_word_importance(model, vectorizer, top_n: int = 10) -> dict:
    feature_names = vectorizer.get_feature_names_out()

    if hasattr(model, "coef_"):
        coefficients = model.coef_[0]
        real_indices = coefficients.argsort()[-top_n:][::-1]
        fake_indices = coefficients.argsort()[:top_n]

        return {
            "top_real_words": [
                {"word": feature_names[i], "weight": float(coefficients[i])} for i in real_indices
            ],
            "top_fake_words": [
                {"word": feature_names[i], "weight": float(coefficients[i])} for i in fake_indices
            ],
        }

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        top_indices = importances.argsort()[-top_n:][::-1]
        generic = [{"word": feature_names[i], "weight": float(importances[i])} for i in top_indices]
        return {"top_real_words": generic, "top_fake_words": generic}

    return {"top_real_words": [], "top_fake_words": []}


def get_local_word_contributions(preprocessed_text: str, model, vectorizer, top_n: int = 10) -> dict:
    if not preprocessed_text:
        return {"top_real_words": [], "top_fake_words": []}

    if not hasattr(model, "coef_"):
        return get_global_word_importance(model, vectorizer, top_n=top_n)

    vectorized = vectorizer.transform([preprocessed_text]).tocsr()
    if vectorized.nnz == 0:
        return {"top_real_words": [], "top_fake_words": []}

    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]

    indices = vectorized.indices
    values = vectorized.data
    contributions = values * coefficients[indices]

    ranked = sorted(
        ((idx, float(score)) for idx, score in zip(indices, contributions)),
        key=lambda item: item[1],
    )
    fake_ranked = ranked[:top_n]
    real_ranked = list(reversed(ranked[-top_n:]))

    return {
        "top_fake_words": [{"word": feature_names[idx], "weight": score} for idx, score in fake_ranked],
        "top_real_words": [{"word": feature_names[idx], "weight": score} for idx, score in real_ranked],
    }


def explain_with_shap(preprocessed_text: str, model, vectorizer, top_n: int = 8) -> dict:
    try:
        import shap  # type: ignore
    except Exception:
        return {"available": False, "reason": "shap not installed", "items": []}

    try:
        vectorized = vectorizer.transform([preprocessed_text])
        if vectorized.nnz == 0:
            return {"available": True, "items": []}

        background = vectorizer.transform(["sample background text"])
        explainer = shap.Explainer(model, background)
        explanation = explainer(vectorized)
        shap_values = explanation.values[0]
        feature_names = vectorizer.get_feature_names_out()

        present = vectorized.indices
        scored = sorted(
            ((idx, float(shap_values[idx])) for idx in present),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:top_n]

        items = [{"word": feature_names[idx], "shap_value": score} for idx, score in scored]
        return {"available": True, "items": items}
    except Exception as exc:
        return {"available": False, "reason": str(exc), "items": []}


def explain_with_lime(raw_text: str, model, vectorizer, top_n: int = 8) -> dict:
    try:
        from lime.lime_text import LimeTextExplainer  # type: ignore
    except Exception:
        return {"available": False, "reason": "lime not installed", "items": []}

    try:
        explainer = LimeTextExplainer(class_names=["FAKE NEWS", "REAL NEWS"])

        def predictor(texts: list[str]):
            processed = [preprocess_text(text) for text in texts]
            return _predict_proba_compat(model, vectorizer.transform(processed))

        explanation = explainer.explain_instance(raw_text, predictor, num_features=top_n)
        items = [{"token": token, "weight": float(weight)} for token, weight in explanation.as_list()]
        return {"available": True, "items": items}
    except Exception as exc:
        return {"available": False, "reason": str(exc), "items": []}


def build_explanation_payload(
    raw_text: str,
    preprocessed_text: str,
    model,
    vectorizer,
    top_n: int = 10,
    include_shap: bool = True,
    include_lime: bool = True,
) -> dict:
    explanation = get_local_word_contributions(preprocessed_text, model, vectorizer, top_n=top_n)
    explanation["global_importance"] = get_global_word_importance(model, vectorizer, top_n=top_n)

    if include_shap:
        explanation["shap"] = explain_with_shap(preprocessed_text, model, vectorizer, top_n=top_n)
    if include_lime:
        explanation["lime"] = explain_with_lime(raw_text, model, vectorizer, top_n=top_n)

    return explanation

