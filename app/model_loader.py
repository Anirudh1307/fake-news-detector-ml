from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock

LOGGER = logging.getLogger(__name__)

model = None
vectorizer = None

_LOAD_LOCK = Lock()
_loaded_model_path: Path | None = None
_loaded_vectorizer_path: Path | None = None


class ArtifactLoadError(RuntimeError):
    """Base error for model/vectorizer loading failures."""


class MissingModelArtifactsError(ArtifactLoadError):
    """Raised when one or more expected artifact files do not exist."""

    def __init__(self, missing_files: list[Path]) -> None:
        self.missing_files = missing_files
        missing = ", ".join(str(path) for path in missing_files)
        super().__init__(
            f"Model artifacts missing: {missing}. "
            "Train locally and deploy artifacts: "
            "python training/train_models.py --fake-path data/Fake.csv "
            "--true-path data/True.csv --models-dir models"
        )


def _normalize_paths(model_path: str | Path, vectorizer_path: str | Path) -> tuple[Path, Path]:
    return Path(model_path), Path(vectorizer_path)


def _cached_paths_match(model_path: Path, vectorizer_path: Path) -> bool:
    return _loaded_model_path == model_path and _loaded_vectorizer_path == vectorizer_path


def missing_artifacts(model_path: str | Path, vectorizer_path: str | Path) -> list[Path]:
    resolved_model_path, resolved_vectorizer_path = _normalize_paths(model_path, vectorizer_path)
    missing: list[Path] = []
    if not resolved_model_path.exists():
        missing.append(resolved_model_path)
    if not resolved_vectorizer_path.exists():
        missing.append(resolved_vectorizer_path)
    return missing


def is_model_loaded(model_path: str | Path | None = None, vectorizer_path: str | Path | None = None) -> bool:
    if model is None or vectorizer is None:
        return False
    if model_path is None or vectorizer_path is None:
        return True
    resolved_model_path, resolved_vectorizer_path = _normalize_paths(model_path, vectorizer_path)
    return _cached_paths_match(resolved_model_path, resolved_vectorizer_path)


def get_model(model_path: str | Path, vectorizer_path: str | Path) -> tuple[object, object]:
    global model, vectorizer, _loaded_model_path, _loaded_vectorizer_path

    resolved_model_path, resolved_vectorizer_path = _normalize_paths(model_path, vectorizer_path)

    if is_model_loaded(resolved_model_path, resolved_vectorizer_path):
        return model, vectorizer

    with _LOAD_LOCK:
        if is_model_loaded(resolved_model_path, resolved_vectorizer_path):
            return model, vectorizer

        if not _cached_paths_match(resolved_model_path, resolved_vectorizer_path):
            model = None
            vectorizer = None
            _loaded_model_path = resolved_model_path
            _loaded_vectorizer_path = resolved_vectorizer_path

        missing_files = missing_artifacts(resolved_model_path, resolved_vectorizer_path)
        if missing_files:
            LOGGER.error(
                "Model artifacts missing. model_path=%s vectorizer_path=%s",
                resolved_model_path,
                resolved_vectorizer_path,
            )
            raise MissingModelArtifactsError(missing_files)

        try:
            import joblib

            if model is None:
                model = joblib.load(resolved_model_path)
            if vectorizer is None:
                vectorizer = joblib.load(resolved_vectorizer_path)
        except Exception as exc:
            model = None
            vectorizer = None
            raise ArtifactLoadError(
                f"Failed to load model artifacts from '{resolved_model_path}' "
                f"and '{resolved_vectorizer_path}': {exc}"
            ) from exc

        LOGGER.info(
            "Model artifacts loaded lazily. model_path=%s vectorizer_path=%s",
            resolved_model_path,
            resolved_vectorizer_path,
        )
        return model, vectorizer


class ModelArtifacts:
    def __init__(self, model_path: str | Path, vectorizer_path: str | Path) -> None:
        self.model_path, self.vectorizer_path = _normalize_paths(model_path, vectorizer_path)

    def missing_files(self) -> list[Path]:
        return missing_artifacts(self.model_path, self.vectorizer_path)

    def ensure_loaded(self) -> None:
        get_model(self.model_path, self.vectorizer_path)

    @property
    def model(self) -> object | None:
        if not _cached_paths_match(self.model_path, self.vectorizer_path):
            return None
        return model

    @property
    def vectorizer(self) -> object | None:
        if not _cached_paths_match(self.model_path, self.vectorizer_path):
            return None
        return vectorizer

    @property
    def is_ready(self) -> bool:
        return is_model_loaded(self.model_path, self.vectorizer_path)


def create_artifact_loader(
    model_path: str | Path,
    vectorizer_path: str | Path,
    metadata_path: str | Path | None = None,
    auto_train: bool | None = None,
) -> ModelArtifacts:
    # Keep signature backward-compatible for existing calls.
    del metadata_path
    del auto_train
    return ModelArtifacts(model_path=model_path, vectorizer_path=vectorizer_path)
