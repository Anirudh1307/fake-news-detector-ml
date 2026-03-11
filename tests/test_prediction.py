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

    assert output["prediction"] in {"FAKE NEWS", "REAL NEWS"}
    assert isinstance(output["confidence"], float)
    assert "explanation" in output
    assert "top_fake_words" in output["explanation"]
    assert "top_real_words" in output["explanation"]

