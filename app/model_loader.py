from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

import joblib


@dataclass
class ModelArtifacts:
    model_path: Path
    vectorizer_path: Path
    metadata_path: Path | None = None
    model: object | None = None
    vectorizer: object | None = None
    metadata: dict = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def load(self) -> None:
        with self._lock:
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
            if self.metadata_path and self.metadata_path.exists():
                self.metadata = joblib.load(self.metadata_path)

    def ensure_loaded(self) -> None:
        if self.model is None or self.vectorizer is None:
            self.load()

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.vectorizer is not None


def create_artifact_loader(
    model_path: str | Path,
    vectorizer_path: str | Path,
    metadata_path: str | Path | None = None,
) -> ModelArtifacts:
    return ModelArtifacts(
        model_path=Path(model_path),
        vectorizer_path=Path(vectorizer_path),
        metadata_path=Path(metadata_path) if metadata_path else None,
    )

