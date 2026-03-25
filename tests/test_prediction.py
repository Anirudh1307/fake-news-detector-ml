import app.utils as utils
from app.utils import run_prediction


def test_run_prediction_returns_expected_fields(test_app):
    artifacts = test_app.extensions["model_artifacts"]
    artifacts.ensure_loaded()

    output = run_prediction(
        raw_text="Official reports confirm the policy outcomes.",
        model=artifacts.model,
        vectorizer=artifacts.vectorizer,
        include_shap=False,
        include_lime=False,
    )

    assert output["prediction"] in {"FAKE", "REAL", "UNCERTAIN", "INSUFFICIENT_CONTEXT"}
    assert isinstance(output["confidence"], int)
    assert "explanation" in output
    assert "top_fake_words" in output["explanation"]
    assert "top_real_words" in output["explanation"]
    assert "hybrid_signals" in output["explanation"]


def test_run_prediction_applies_strong_fake_rule_before_ml(test_app):
    artifacts = test_app.extensions["model_artifacts"]
    artifacts.ensure_loaded()

    output = run_prediction(
        raw_text="A miracle cure promises teleport travel and time travel without evidence or testing.",
        model=artifacts.model,
        vectorizer=artifacts.vectorizer,
        include_shap=False,
        include_lime=False,
    )

    assert output["prediction"] == "FAKE"
    assert output["confidence"] == 85
    assert output["explanation"]["hybrid_signals"]["strong_fake_match"]


def test_run_prediction_returns_uncertain_for_vague_text(test_app):
    artifacts = test_app.extensions["model_artifacts"]
    artifacts.ensure_loaded()

    output = run_prediction(
        raw_text="Some say the situation is changing fast and details emerging while nothing confirmed yet.",
        model=artifacts.model,
        vectorizer=artifacts.vectorizer,
        include_shap=False,
        include_lime=False,
    )

    assert output["prediction"] == "UNCERTAIN"
    assert output["confidence"] == 45
    assert output["explanation"]["hybrid_signals"]["vague_detected"] is True


def test_run_prediction_returns_real_for_low_confidence_neutral_text(test_app, monkeypatch):
    artifacts = test_app.extensions["model_artifacts"]
    artifacts.ensure_loaded()

    monkeypatch.setattr(utils, "_ml_scores", lambda model, vectorizer, preprocessed_text: (0.5, 0.5))

    output = run_prediction(
        raw_text="Inflation moved across regions over the quarter with changes in prices and spending.",
        model=artifacts.model,
        vectorizer=artifacts.vectorizer,
        include_shap=False,
        include_lime=False,
    )

    assert output["prediction"] == "REAL"
    assert output["confidence"] == 40


def test_run_prediction_returns_uncertain_for_low_confidence_text_with_fake_keywords(test_app, monkeypatch):
    artifacts = test_app.extensions["model_artifacts"]
    artifacts.ensure_loaded()

    monkeypatch.setattr(utils, "_ml_scores", lambda model, vectorizer, preprocessed_text: (0.5, 0.5))

    output = run_prediction(
        raw_text="Breaking exposed claims are circulating online without clarity.",
        model=artifacts.model,
        vectorizer=artifacts.vectorizer,
        include_shap=False,
        include_lime=False,
    )

    assert output["prediction"] == "UNCERTAIN"
    assert output["confidence"] == 40
