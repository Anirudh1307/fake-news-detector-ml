from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any, Callable

from flask import Flask, jsonify, render_template, request

from app.model_loader import (
    ArtifactLoadError,
    MissingModelArtifactsError,
    create_artifact_loader,
    get_model,
    is_model_loaded,
    missing_artifacts,
)
from dashboard.analytics import build_analytics_summary

BASE_DIR = Path(__file__).resolve().parents[1]


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_artifact_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    model_path = Path(config.get("MODEL_PATH", BASE_DIR / "models" / "best_model.joblib"))
    vectorizer_path = Path(config.get("VECTORIZER_PATH", BASE_DIR / "models" / "tfidf_vectorizer.joblib"))
    return model_path, vectorizer_path


def _resolve_explainability_flags(
    app: Flask,
    include_shap: bool,
    include_lime: bool,
) -> tuple[bool, bool]:
    shap_enabled = _as_bool(app.config.get("ENABLE_SHAP"), default=False)
    lime_enabled = _as_bool(app.config.get("ENABLE_LIME"), default=False)
    return include_shap and shap_enabled, include_lime and lime_enabled


def _artifact_error_response(app: Flask, exc: ArtifactLoadError):
    model_path, vectorizer_path = _resolve_artifact_paths(app.config)
    missing = [str(path) for path in missing_artifacts(model_path, vectorizer_path)]
    error_code = "MODEL_ARTIFACTS_MISSING" if missing else "MODEL_ARTIFACT_LOAD_FAILED"
    if isinstance(exc, MissingModelArtifactsError):
        error_code = "MODEL_ARTIFACTS_MISSING"
    return (
        jsonify(
            {
                "error": str(exc),
                "error_code": error_code,
                "model_loaded": is_model_loaded(model_path, vectorizer_path),
                "model_path": str(model_path),
                "vectorizer_path": str(vectorizer_path),
                "missing_artifacts": missing,
            }
        ),
        503,
    )


def _append_jsonl(log_path: str | Path, payload: dict[str, Any]) -> None:
    from app.utils import append_jsonl

    append_jsonl(log_path, payload)


def _fetch_article_text(url: str) -> str | dict[str, Any]:
    from app.utils import fetch_article_text

    return fetch_article_text(url)


def _preprocess_and_count_tokens(text: str) -> tuple[str, int]:
    from app.preprocessing import preprocess_and_count_tokens

    return preprocess_and_count_tokens(text)


def _read_jsonl(log_path: str | Path) -> list[dict[str, Any]]:
    from app.utils import read_jsonl

    return read_jsonl(log_path)


def _run_prediction(**kwargs: Any) -> dict[str, Any]:
    from app.utils import run_prediction

    return run_prediction(**kwargs)


def _predict_payload(
    app: Flask,
    text: str,
    source: str,
    source_url: str | None = None,
    preprocessed_text: str | None = None,
    include_shap: bool = False,
    include_lime: bool = False,
) -> dict:
    model_path, vectorizer_path = _resolve_artifact_paths(app.config)
    model, vectorizer = get_model(model_path, vectorizer_path)

    result = _run_prediction(
        raw_text=text,
        preprocessed_text=preprocessed_text,
        model=model,
        vectorizer=vectorizer,
        include_shap=include_shap,
        include_lime=include_lime,
        source_url=source_url,
    )

    response = {
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "top_fake_words": result["explanation"].get("top_fake_words", []),
        "top_real_words": result["explanation"].get("top_real_words", []),
        "explanation": result["explanation"],
    }

    log_payload = {
        "source": source,
        "url": source_url,
        "prediction": response["prediction"],
        "confidence": response["confidence"],
        "explanation": {
            "top_fake_words": response["top_fake_words"],
            "top_real_words": response["top_real_words"],
        },
    }
    _append_jsonl(app.config["ANALYTICS_LOG_PATH"], log_payload)
    return response


def create_app(config: dict[str, Any] | None = None) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(BASE_DIR / "templates"),
    )
    app.config.update(
        MODEL_PATH=os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "best_model.joblib")),
        VECTORIZER_PATH=os.getenv("VECTORIZER_PATH", str(BASE_DIR / "models" / "tfidf_vectorizer.joblib")),
        GNEWS_API_KEY=os.getenv("GNEWS_API_KEY", ""),
        GOOGLE_FACTCHECK_API_KEY=os.getenv("GOOGLE_FACTCHECK_API_KEY", ""),
        ANALYTICS_LOG_PATH=os.getenv(
            "ANALYTICS_LOG_PATH",
            str(Path(tempfile.gettempdir()) / "prediction_logs.jsonl"),
        ),
        ENABLE_SHAP=_as_bool(os.getenv("ENABLE_SHAP"), default=False),
        ENABLE_LIME=_as_bool(os.getenv("ENABLE_LIME"), default=False),
        MAX_INPUT_CHARS=_as_int(os.getenv("MAX_INPUT_CHARS"), default=30000),
    )
    if config:
        app.config.update(config)

    model_path, vectorizer_path = _resolve_artifact_paths(app.config)
    app.extensions["model_artifacts"] = create_artifact_loader(model_path, vectorizer_path)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/dashboard")
    def dashboard():
        return render_template("dashboard.html")

    @app.get("/health")
    def health():
        model_path, vectorizer_path = _resolve_artifact_paths(app.config)
        missing = [str(path) for path in missing_artifacts(model_path, vectorizer_path)]
        return jsonify(
            {
                "status": "ok",
                "model_loaded": is_model_loaded(model_path, vectorizer_path),
                "model_files_present": not missing,
                "missing_artifacts": missing,
                "model_path": str(model_path),
                "vectorizer_path": str(vectorizer_path),
                "enable_shap": _as_bool(app.config.get("ENABLE_SHAP"), default=False),
                "enable_lime": _as_bool(app.config.get("ENABLE_LIME"), default=False),
            }
        )

    @app.get("/api/analytics")
    def analytics():
        records = _read_jsonl(app.config["ANALYTICS_LOG_PATH"])
        return jsonify(build_analytics_summary(records))

    @app.post("/predict")
    def predict():
        data = request.get_json(silent=True) or {}
        text = data.get("text", "")
        include_shap, include_lime = _resolve_explainability_flags(
            app,
            include_shap=_as_bool(data.get("include_shap"), default=False),
            include_lime=_as_bool(data.get("include_lime"), default=False),
        )

        if not isinstance(text, str) or not text.strip():
            return jsonify({"error": "Input text is required."}), 400
        if len(text) > app.config["MAX_INPUT_CHARS"]:
            return jsonify({"error": f"Input text too long. Max {app.config['MAX_INPUT_CHARS']} characters."}), 400

        original_word_count = len(text.split())
        if original_word_count < 12:
            return jsonify(
                {
                    "prediction": "INSUFFICIENT_CONTEXT",
                    "confidence": 0,
                    "message": "Input text is too short for reliable classification.",
                }
            )
        preprocessed_text, _ = _preprocess_and_count_tokens(text)

        try:
            response = _predict_payload(
                app,
                text=text,
                source="text",
                preprocessed_text=preprocessed_text,
                include_shap=include_shap,
                include_lime=include_lime,
            )
            return jsonify(response)
        except ArtifactLoadError as exc:
            return _artifact_error_response(app, exc)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": f"Prediction failed: {exc}"}), 500

    @app.post("/analyze_url")
    def analyze_url():
        data = request.get_json(silent=True) or {}
        url = data.get("url", "")
        include_shap, include_lime = _resolve_explainability_flags(
            app,
            include_shap=_as_bool(data.get("include_shap"), default=False),
            include_lime=_as_bool(data.get("include_lime"), default=False),
        )

        if not isinstance(url, str) or not url.strip():
            return jsonify({"error": "A valid URL is required."}), 400

        article_fetcher: Callable[[str], str | dict[str, Any]] = app.config.get(
            "ARTICLE_FETCHER",
            _fetch_article_text,
        )

        try:
            article_result = article_fetcher(url)
            if isinstance(article_result, dict):
                status_code = _as_int(article_result.get("status_code"), default=400)
                return jsonify({"error": article_result.get("error", "Unable to analyze URL.")}), status_code

            article_text = article_result
            if not isinstance(article_text, str) or not article_text.strip():
                return jsonify({"error": "Unable to extract article text from this URL."}), 400

            if len(article_text) > app.config["MAX_INPUT_CHARS"]:
                return (
                    jsonify(
                        {
                            "error": (
                                f"Article text too long after extraction. "
                                f"Max {app.config['MAX_INPUT_CHARS']} characters."
                            )
                        }
                    ),
                    400,
                )
            prediction = _predict_payload(
                app,
                text=article_text,
                source="url",
                source_url=url,
                include_shap=include_shap,
                include_lime=include_lime,
            )
            return jsonify(
                {
                    "url": url,
                    "article_preview": article_text[:350],
                    "article_char_count": len(article_text),
                    **prediction,
                }
            )
        except ArtifactLoadError as exc:
            return _artifact_error_response(app, exc)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": f"URL analysis failed: {exc}"}), 500

    return app
