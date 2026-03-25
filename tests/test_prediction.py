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
