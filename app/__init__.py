"""Application package for Fake News Detector."""

import logging
import os
from pathlib import Path

from app.model_loader import ensure_model_exists
from app.routes import create_app

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "best_model.joblib"))
VECTORIZER_PATH = Path(os.getenv("VECTORIZER_PATH", BASE_DIR / "models" / "tfidf_vectorizer.joblib"))
AUTO_TRAIN_ON_BOOT = os.getenv("AUTO_TRAIN_ON_BOOT", "0").strip().lower() in {"1", "true", "yes", "on"}
AUTO_TRAIN_ON_REQUEST = os.getenv("AUTO_TRAIN_ON_REQUEST", "0").strip().lower() in {"1", "true", "yes", "on"}

# Startup check for Render/Docker cold start.
ensure_model_exists(
    MODEL_PATH,
    VECTORIZER_PATH,
    blocking=False,
    allow_training=AUTO_TRAIN_ON_BOOT,
)

app = create_app(
    {
        "MODEL_PATH": str(MODEL_PATH),
        "VECTORIZER_PATH": str(VECTORIZER_PATH),
        "AUTO_TRAIN_ON_REQUEST": AUTO_TRAIN_ON_REQUEST,
    }
)

__all__ = ["app", "create_app"]
