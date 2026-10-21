"""
app/ml/predict.py
-----------------
Inference engine for 5Star Predictions.

This module exposes two classes — ``ModelLoader`` and ``PredictionEngine`` —
plus module-level singletons and a ``load_models()`` startup hook called by
the FastAPI lifespan event.

Typical call chain
------------------
  app startup  →  load_models()
  request      →  prediction_engine.predict(feature_dict)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

from app.ml.features import (
    FT_OU_FEATURE_COLUMNS,
    HT_OU_FEATURE_COLUMNS,
    WINNER_FEATURE_COLUMNS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_MODELS_DIR = Path(__file__).resolve().parent / "models"
_WINNER_MODEL_PATH = _MODELS_DIR / "winner_model.json"
_FT_OU_MODEL_PATH = _MODELS_DIR / "ft_ou_model.json"
_HT_OU_MODEL_PATH = _MODELS_DIR / "ht_ou_model.json"
_METADATA_PATH = _MODELS_DIR / "metadata.json"


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class Calibrator:
    """
    Piecewise-linear isotonic map from raw model probability to calibrated
    probability, fitted on a held-out split by train.py.  An empty map
    leaves probabilities unchanged.
    """

    def __init__(self, spec: dict | None) -> None:
        spec = spec or {}
        self._x = np.asarray(spec.get("x", []), dtype=float)
        self._y = np.asarray(spec.get("y", []), dtype=float)

    def __call__(self, p: float) -> float:
        if self._x.size == 0:
            return p
        return float(np.interp(p, self._x, self._y))


# ---------------------------------------------------------------------------
# ModelLoader
# ---------------------------------------------------------------------------

class ModelLoader:
    """
    Responsible for loading XGBoost model files from disk.

    The winner and full-time O/U models are required.  The half-time model
    is loaded only when metadata marks it as validated, so an unvalidated
    model is never served.
    """

    def __init__(self) -> None:
        self._winner_model: XGBClassifier | None = None
        self._ft_ou_model: XGBClassifier | None = None
        self._ht_ou_model: XGBClassifier | None = None
        self._calibrators: dict[str, Calibrator] = {}
        self._metadata: dict = {}
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load the models from ``app/ml/models/``.

        Raises
        ------
        FileNotFoundError
            If a required model file does not exist.  Run
            ``python -m app.ml.train`` first to produce the model artefacts.
        """
        for path in (_WINNER_MODEL_PATH, _FT_OU_MODEL_PATH):
            if not path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {path}\n"
                    "Run `python -m app.ml.train` to produce model artefacts."
                )

        metadata: dict = {}
        if _METADATA_PATH.exists():
            metadata = json.loads(_METADATA_PATH.read_text(encoding="utf-8"))
            logger.info(
                "Model metadata: version=%s trained_at=%s",
                metadata.get("version"),
                metadata.get("trained_at"),
            )
        else:
            logger.warning("metadata.json not found; version info will be unavailable.")
        model_meta: dict = metadata.get("models", {})

        # Build everything first, then swap in, so a retrain that fails
        # half-way never leaves a mix of old and new models.
        winner_model = XGBClassifier()
        winner_model.load_model(str(_WINNER_MODEL_PATH))
        ft_ou_model = XGBClassifier()
        ft_ou_model.load_model(str(_FT_OU_MODEL_PATH))

        ht_ou_model: XGBClassifier | None = None
        if model_meta.get("ht_ou_model", {}).get("validated") and _HT_OU_MODEL_PATH.exists():
            ht_ou_model = XGBClassifier()
            ht_ou_model.load_model(str(_HT_OU_MODEL_PATH))
        else:
            logger.info("Half-time O/U model is not validated; HT picks are disabled.")

        self._winner_model = winner_model
        self._ft_ou_model = ft_ou_model
        self._ht_ou_model = ht_ou_model
        self._calibrators = {
            name: Calibrator(meta.get("calibration")) for name, meta in model_meta.items()
        }
        self._metadata = metadata
        self._loaded = True
        logger.info("Models loaded successfully.")

    @property
    def is_loaded(self) -> bool:
        """Return ``True`` once ``load()`` has completed successfully."""
        return self._loaded

    # ------------------------------------------------------------------
    # Model accessors (package-internal)
    # ------------------------------------------------------------------

    @property
    def winner_model(self) -> XGBClassifier:
        self._assert_loaded()
        return self._winner_model  # type: ignore[return-value]

    @property
    def ft_ou_model(self) -> XGBClassifier:
        self._assert_loaded()
        return self._ft_ou_model  # type: ignore[return-value]

    @property
    def ht_ou_model(self) -> XGBClassifier | None:
        """The validated half-time model, or None when HT picks are disabled."""
        self._assert_loaded()
        return self._ht_ou_model

    def calibrator(self, model_name: str) -> Calibrator:
        return self._calibrators.get(model_name) or Calibrator(None)

    @property
    def model_version(self) -> str:
        """Return the version string from metadata, or 'unknown'."""
        return self._metadata.get("version", "unknown")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assert_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(
                "Models have not been loaded. Call load_models() at application startup."
            )


# ---------------------------------------------------------------------------
# PredictionEngine
# ---------------------------------------------------------------------------

class PredictionEngine:
    """
    Runs predictions against the loaded XGBoost models.

    Parameters
    ----------
    loader:
        A ``ModelLoader`` instance that has already called ``load()``.
    """

    def __init__(self, loader: ModelLoader) -> None:
        self._loader = loader

    def _binary_pick(
        self,
        model: XGBClassifier,
        model_name: str,
        feature_dict: dict,
        columns: list[str],
        labels: tuple[str, str],
    ) -> tuple[str, float]:
        """Return (label, trust %) where labels = (class 0 label, class 1 label)."""
        try:
            X = np.array([[float(feature_dict[col]) for col in columns]])
        except KeyError as exc:
            raise KeyError(f"Missing feature column in feature_dict: {exc}.") from exc
        raw_p1 = float(model.predict_proba(X)[0][1])
        p1 = self._loader.calibrator(model_name)(raw_p1)
        pick = labels[1] if p1 >= 0.5 else labels[0]
        return pick, round(max(p1, 1.0 - p1) * 100, 1)

    def predict(self, feature_dict: dict) -> dict:
        """
        Run the models for the given feature dictionary.

        An O/U head is skipped (pick and trust are ``None``) when its
        sportsbook line is missing, because the models were only trained on
        games with a real line.  The half-time head is also skipped while
        no validated half-time model is loaded.

        Returns
        -------
        dict with keys:
            winner_pick     : 'Home' | 'Away'
            winner_trust    : float — calibrated confidence % (50–100)
            ft_ou_pick      : 'Over' | 'Under' | None
            ft_ou_trust     : float | None
            ht_ou_pick      : 'Over' | 'Under' | None
            ht_ou_trust     : float | None
            model_version   : str

        Raises
        ------
        RuntimeError
            If ``ModelLoader.load()`` has not been called.
        KeyError
            If a required feature column is absent from ``feature_dict``.
        """
        if not self._loader.is_loaded:
            raise RuntimeError(
                "PredictionEngine: models are not loaded. "
                "Ensure load_models() is called at application startup."
            )

        winner_pick, winner_trust = self._binary_pick(
            self._loader.winner_model,
            "winner_model",
            feature_dict,
            WINNER_FEATURE_COLUMNS,
            ("Away", "Home"),
        )

        ft_ou_pick = ft_ou_trust = None
        if feature_dict.get("ou_line") is not None:
            ft_ou_pick, ft_ou_trust = self._binary_pick(
                self._loader.ft_ou_model,
                "ft_ou_model",
                feature_dict,
                FT_OU_FEATURE_COLUMNS,
                ("Under", "Over"),
            )

        ht_ou_pick = ht_ou_trust = None
        ht_model = self._loader.ht_ou_model
        if (
            ht_model is not None
            and feature_dict.get("ou_line") is not None
            and feature_dict.get("ht_ou_line") is not None
        ):
            ht_ou_pick, ht_ou_trust = self._binary_pick(
                ht_model,
                "ht_ou_model",
                feature_dict,
                HT_OU_FEATURE_COLUMNS,
                ("Under", "Over"),
            )

        return {
            "winner_pick": winner_pick,
            "winner_trust": winner_trust,
            "ft_ou_pick": ft_ou_pick,
            "ft_ou_trust": ft_ou_trust,
            "ht_ou_pick": ht_ou_pick,
            "ht_ou_trust": ht_ou_trust,
            "model_version": self._loader.model_version,
        }


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

#: Shared ModelLoader instance — populated by ``load_models()`` at startup.
model_loader = ModelLoader()

#: Shared PredictionEngine — ready to use after ``load_models()`` succeeds.
prediction_engine = PredictionEngine(model_loader)


# ---------------------------------------------------------------------------
# Startup hook
# ---------------------------------------------------------------------------

def load_models() -> None:
    """
    Load the XGBoost models into the shared ``model_loader`` singleton.

    Logs a warning (rather than raising) if model files are absent so that
    the API can still start in environments where training has not yet run.
    """
    try:
        model_loader.load()
        logger.info("5Star Predictions models are ready.")
    except FileNotFoundError as exc:
        logger.warning(
            "Could not load ML models: %s. "
            "Prediction endpoints will be unavailable until models are trained.",
            exc,
        )
