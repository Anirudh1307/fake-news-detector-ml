from __future__ import annotations

import numpy as np


class CalibratedHybridModel:
    """Inference wrapper that preserves linear-model attributes after calibration."""

    def __init__(self, base_model, calibrator) -> None:
        self.base_model = base_model
        self.calibrator = calibrator
        self.classes_ = getattr(calibrator, "classes_", None)

    def predict(self, X):
        return self.calibrator.predict(X)

    def predict_proba(self, X):
        return self.calibrator.predict_proba(X)

    def decision_function(self, X):
        if hasattr(self.base_model, "decision_function"):
            return self.base_model.decision_function(X)
        probs = self.predict_proba(X)[:, 1]
        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
        return np.log(probs / (1.0 - probs))

    @property
    def coef_(self):
        base = self.__dict__.get("base_model")
        if base is None:
            raise AttributeError("coef_")
        return base.coef_

    def __getattr__(self, name: str):
        if name in {"base_model", "calibrator"}:
            raise AttributeError(name)
        base = self.__dict__.get("base_model")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)
