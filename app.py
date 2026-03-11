import logging
import os
from pathlib import Path

from app.routes import create_app

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "best_model.joblib"))
VECTORIZER_PATH = Path(os.getenv("VECTORIZER_PATH", BASE_DIR / "models" / "tfidf_vectorizer.joblib"))

app = create_app(
    {
        "MODEL_PATH": str(MODEL_PATH),
        "VECTORIZER_PATH": str(VECTORIZER_PATH),
    }
)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=False,
    )

