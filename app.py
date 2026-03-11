import os
from pathlib import Path

from app.model_loader import check_or_train_model
from app.routes import create_app

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "best_model.joblib"))
VECTORIZER_PATH = Path(os.getenv("VECTORIZER_PATH", BASE_DIR / "models" / "tfidf_vectorizer.joblib"))


def build_app():
    # Startup model check before serving requests.
    check_or_train_model(MODEL_PATH, VECTORIZER_PATH)
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
