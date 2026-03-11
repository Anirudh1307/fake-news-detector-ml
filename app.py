import os
from pathlib import Path
import logging

from app.model_loader import ensure_model_exists
from app.routes import create_app

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "best_model.joblib"))
VECTORIZER_PATH = Path(os.getenv("VECTORIZER_PATH", BASE_DIR / "models" / "tfidf_vectorizer.joblib"))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)


def build_app():
    # Startup model check before serving requests.
    ensure_model_exists(MODEL_PATH, VECTORIZER_PATH)
    return create_app(
        {
            "MODEL_PATH": str(MODEL_PATH),
            "VECTORIZER_PATH": str(VECTORIZER_PATH),
        }
    )


app = build_app()

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=False,
    )
