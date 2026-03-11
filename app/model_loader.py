from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock, Thread
from typing import Any

import joblib

LOGGER = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_SCRIPT = BASE_DIR / "training" / "train_models.py"
_TRAINING_THREAD: Thread | None = None
_TRAINING_STATE_LOCK = Lock()
_LAST_TRAINING_ERROR: str | None = None


def is_training_in_progress() -> bool:
    with _TRAINING_STATE_LOCK:
        return _TRAINING_THREAD is not None and _TRAINING_THREAD.is_alive()


def get_last_training_error() -> str | None:
    with _TRAINING_STATE_LOCK:
        return _LAST_TRAINING_ERROR


def _set_last_training_error(message: str | None) -> None:
    global _LAST_TRAINING_ERROR
    with _TRAINING_STATE_LOCK:
        _LAST_TRAINING_ERROR = message


def _run_training_subprocess(
    command: list[str],
    work_dir: Path,
    model_path: Path,
    vectorizer_path: Path,
) -> bool:
    try:
        result = subprocess.run(
            command,
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        LOGGER.exception("Failed to start model training process.")
        _set_last_training_error("Failed to start model training process.")
        return False

    if result.stdout:
        LOGGER.info(result.stdout.strip())
    if result.stderr:
        LOGGER.warning(result.stderr.strip())

    if result.returncode != 0:
        error_message = f"Model training failed with exit code {result.returncode}"
        LOGGER.error(error_message)
        _set_last_training_error(error_message)
        return False

    LOGGER.info("Model training complete.")
    if model_path.exists() and vectorizer_path.exists():
        LOGGER.info("Model successfully trained and saved.")

    _set_last_training_error(None)
    return True


def _start_background_training(
    command: list[str],
    work_dir: Path,
    model_path: Path,
    vectorizer_path: Path,
) -> None:
    global _TRAINING_THREAD

    def _target():
        _run_training_subprocess(command, work_dir, model_path=model_path, vectorizer_path=vectorizer_path)

    with _TRAINING_STATE_LOCK:
        if _TRAINING_THREAD is not None and _TRAINING_THREAD.is_alive():
            return
        _TRAINING_THREAD = Thread(target=_target, daemon=True, name="model-training-thread")
        _TRAINING_THREAD.start()


def ensure_model_exists(
    model_path: str | Path,
    vectorizer_path: str | Path,
    train_script_path: str | Path | None = None,
    base_dir: str | Path | None = None,
    blocking: bool = False,
    allow_training: bool = True,
) -> bool:
    """Ensure model artifacts exist, training them if missing."""
    model_path = Path(model_path)
    vectorizer_path = Path(vectorizer_path)
    train_script = Path(train_script_path) if train_script_path else DEFAULT_TRAIN_SCRIPT
    work_dir = Path(base_dir) if base_dir else BASE_DIR

    if model_path.exists() and vectorizer_path.exists():
        return True

    if not allow_training:
        message = (
            f"Model artifacts are missing at {model_path} and auto-training is disabled."
        )
        LOGGER.error(message)
        _set_last_training_error(message)
        return False

    LOGGER.warning("Model not found. Training model...")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(train_script),
        "--models-dir",
        str(model_path.parent),
    ]
    if not blocking:
        _start_background_training(
            command,
            work_dir,
            model_path=model_path,
            vectorizer_path=vectorizer_path,
        )
        return False

    trained = _run_training_subprocess(
        command,
        work_dir,
        model_path=model_path,
        vectorizer_path=vectorizer_path,
    )
    if trained and model_path.exists() and vectorizer_path.exists():
        return True

    LOGGER.error("Training finished but artifacts are still missing.")
    return False


def check_or_train_model(
    model_path: str | Path,
    vectorizer_path: str | Path,
    train_script_path: str | Path | None = None,
    base_dir: str | Path | None = None,
) -> bool:
    """Backward-compatible alias."""
    return ensure_model_exists(
        model_path=model_path,
        vectorizer_path=vectorizer_path,
        train_script_path=train_script_path,
        base_dir=base_dir,
        blocking=True,
        allow_training=True,
    )


def is_runtime_auto_train_enabled() -> bool:
    value = os.getenv("AUTO_TRAIN_ON_REQUEST", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


@dataclass
class ModelArtifacts:
    model_path: Path
    vectorizer_path: Path
    metadata_path: Path | None = None
    auto_train: bool = False
    model: object | None = None
    vectorizer: object | None = None
    metadata: dict = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def _load_no_lock(self) -> None:
        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)
        if self.metadata_path and self.metadata_path.exists():
            self.metadata = joblib.load(self.metadata_path)

    def load(self) -> None:
        with self._lock:
            self._load_no_lock()

    def ensure_loaded(self) -> None:
        with self._lock:
            if self.model is not None and self.vectorizer is not None:
                return

            if not (self.model_path.exists() and self.vectorizer_path.exists()):
                trained = ensure_model_exists(
                    self.model_path,
                    self.vectorizer_path,
                    blocking=False,
                    allow_training=self.auto_train,
                )
                if not trained:
                    LOGGER.error("Model artifacts are still unavailable after training attempt.")
                    return

            try:
                self._load_no_lock()
            except FileNotFoundError:
                LOGGER.exception("Model artifacts not found while loading.")
            except Exception:
                LOGGER.exception("Unexpected failure while loading model artifacts.")

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.vectorizer is not None


def create_artifact_loader(
    model_path: str | Path,
    vectorizer_path: str | Path,
    metadata_path: str | Path | None = None,
    auto_train: bool | None = None,
) -> ModelArtifacts:
    if auto_train is None:
        auto_train = is_runtime_auto_train_enabled()
    return ModelArtifacts(
        model_path=Path(model_path),
        vectorizer_path=Path(vectorizer_path),
        metadata_path=Path(metadata_path) if metadata_path else None,
        auto_train=auto_train,
    )
