from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any, Callable

from flask import Flask, jsonify, render_template, request

from app.model_loader import (
    ModelArtifacts,
    create_artifact_loader,
    get_last_training_error,
    is_training_in_progress,
)
from app.utils import append_jsonl, fetch_article_text, read_jsonl, run_prediction
from dashboard.analytics import build_analytics_summary

BASE_DIR = Path(__file__).resolve().parents[1]


def _as_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _resolve_artifact_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    model_path = Path(config.get("MODEL_PATH", BASE_DIR / "models" / "best_model.joblib"))
    vectorizer_path = Path(config.get("VECTORIZER_PATH", BASE_DIR / "models" / "tfidf_vectorizer.joblib"))

    if model_path.exists() and vectorizer_path.exists():
        return model_path, vectorizer_path

    # Backward compatibility with legacy artifact names.
    legacy_model = BASE_DIR / "logistic_regression_model.pkl"
    legacy_vectorizer = BASE_DIR / "tfidf_vectorizer.pkl"
    if legacy_model.exists() and legacy_vectorizer.exists():
        return legacy_model, legacy_vectorizer

    return model_path, vectorizer_path


def _get_artifacts(app: Flask) -> ModelArtifacts:
    artifacts = app.extensions.get("model_artifacts")
    if artifacts is None:
        model_path, vectorizer_path = _resolve_artifact_paths(app.config)
        artifacts = create_artifact_loader(model_path, vectorizer_path)
        app.extensions["model_artifacts"] = artifacts
    return artifacts


def _predict_payload(
    app: Flask,
    text: str,
    source: str,
    source_url: str | None = None,
    include_shap: bool = True,
    include_lime: bool = True,
) -> dict:
    artifacts = _get_artifacts(app)
    artifacts.ensure_loaded()
    if not artifacts.is_ready:
        if is_training_in_progress():
            raise RuntimeError("Model is being trained. Please retry in a minute.")
        last_error = get_last_training_error()
        if last_error:
            raise RuntimeError(f"Model training failed: {last_error}")
        raise RuntimeError("Model is not ready yet. Training may still be running. Try again shortly.")

    result = run_prediction(
        raw_text=text,
        model=artifacts.model,
        vectorizer=artifacts.vectorizer,
        include_shap=include_shap,
        include_lime=include_lime,
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
    append_jsonl(app.config["ANALYTICS_LOG_PATH"], log_payload)

    return response


def create_app(config: dict[str, Any] | None = None) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(BASE_DIR / "templates"),
    )
    app.config.update(
        MODEL_PATH=os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "best_model.joblib")),
        VECTORIZER_PATH=os.getenv("VECTORIZER_PATH", str(BASE_DIR / "models" / "tfidf_vectorizer.joblib")),
        # Render filesystem is ephemeral; this path is safe for runtime write operations.
        ANALYTICS_LOG_PATH=os.getenv(
            "ANALYTICS_LOG_PATH",
            str(Path(tempfile.gettempdir()) / "prediction_logs.jsonl"),
        ),
    )
    if config:
        app.config.update(config)
    _get_artifacts(app)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/dashboard")
    def dashboard():
        return render_template("dashboard.html")

    @app.get("/health")
    def health():
        artifacts = _get_artifacts(app)
        return jsonify(
            {
                "status": "ok",
                "model_loaded": artifacts.is_ready,
                "model_path": str(artifacts.model_path),
                "vectorizer_path": str(artifacts.vectorizer_path),
            }
        )

    @app.get("/api/analytics")
    def analytics():
        records = read_jsonl(app.config["ANALYTICS_LOG_PATH"])
        return jsonify(build_analytics_summary(records))

    @app.post("/predict")
    def predict():
        data = request.get_json(silent=True) or {}
        text = data.get("text", "")
        include_shap = _as_bool(data.get("include_shap"), default=True)
        include_lime = _as_bool(data.get("include_lime"), default=True)

        if not isinstance(text, str) or not text.strip():
            return jsonify({"error": "Input text is required."}), 400

        try:
            response = _predict_payload(
                app,
                text=text,
                source="text",
                include_shap=include_shap,
                include_lime=include_lime,
            )
            return jsonify(response)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 503
        except FileNotFoundError as exc:
            return jsonify({"error": f"Model artifacts not found: {exc}"}), 500
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": f"Prediction failed: {exc}"}), 500

    @app.post("/analyze_url")
    def analyze_url():
        data = request.get_json(silent=True) or {}
        url = data.get("url", "")
        include_shap = _as_bool(data.get("include_shap"), default=True)
        include_lime = _as_bool(data.get("include_lime"), default=True)

        if not isinstance(url, str) or not url.strip():
            return jsonify({"error": "A valid URL is required."}), 400

        article_fetcher: Callable[[str], str] = app.config.get("ARTICLE_FETCHER", fetch_article_text)

        try:
            article_text = article_fetcher(url)
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
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 503
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": f"URL analysis failed: {exc}"}), 500

    return app
