from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from threading import Lock

import joblib

LOGGER = logging.getLogger(__name__)

_SINGLETON_LOCK = Lock()
_SINGLETON: "ModelArtifacts | None" = None


@dataclass
class ModelArtifacts:
    model_path: Path
    vectorizer_path: Path
    model: object | None = None
    vectorizer: object | None = None
    _lock: Lock = field(default_factory=Lock)

    def ensure_loaded(self) -> None:
        """Lazy-load model artifacts once and reuse for all requests."""
        if self.is_ready:
            return

        with self._lock:
            if self.is_ready:
                return

            missing_files: list[str] = []
            if not self.model_path.exists():
                missing_files.append(str(self.model_path))
            if not self.vectorizer_path.exists():
                missing_files.append(str(self.vectorizer_path))

            if missing_files:
                missing = ", ".join(missing_files)
                raise FileNotFoundError(
                    f"Model artifacts missing: {missing}. "
                    "Train locally and deploy artifacts: "
                    "python training/train_models.py --fake-path data/Fake.csv "
                    "--true-path data/True.csv --models-dir models"
                )

            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
            LOGGER.info("Model loaded successfully")

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.vectorizer is not None


def create_artifact_loader(
    model_path: str | Path,
    vectorizer_path: str | Path,
    metadata_path: str | Path | None = None,
    auto_train: bool | None = None,
) -> ModelArtifacts:
    # Keep signature backward-compatible for existing calls.
    del metadata_path
    del auto_train

    global _SINGLETON
    model_path = Path(model_path)
    vectorizer_path = Path(vectorizer_path)

    with _SINGLETON_LOCK:
        if _SINGLETON is None:
            _SINGLETON = ModelArtifacts(model_path=model_path, vectorizer_path=vectorizer_path)
        elif _SINGLETON.model_path != model_path or _SINGLETON.vectorizer_path != vectorizer_path:
            _SINGLETON = ModelArtifacts(model_path=model_path, vectorizer_path=vectorizer_path)
        return _SINGLETON

