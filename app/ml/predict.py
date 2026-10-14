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

from app.ml.features import FEATURE_COLUMNS

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
# ModelLoader
# ---------------------------------------------------------------------------

class ModelLoader:
    """
    Responsible for loading XGBoost model files from disk.

    All three models are loaded into memory via ``load()``.  The
    ``is_loaded`` property can be used to guard inference calls at
    application startup.
    """

    def __init__(self) -> None:
        self._winner_model: XGBClassifier | None = None
        self._ft_ou_model: XGBClassifier | None = None
        self._ht_ou_model: XGBClassifier | None = None
        self._metadata: dict = {}
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load all three XGBoost models from ``app/ml/models/``.

        Raises
        ------
        FileNotFoundError
            If any model file does not exist.  Run ``python -m app.ml.train``
            first to produce the model artefacts.
        """
        for path in (_WINNER_MODEL_PATH, _FT_OU_MODEL_PATH, _HT_OU_MODEL_PATH):
            if not path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {path}\n"
                    "Run `python -m app.ml.train` to produce model artefacts."
                )

        logger.info("Loading winner_model …")
        self._winner_model = XGBClassifier()
        self._winner_model.load_model(str(_WINNER_MODEL_PATH))

        logger.info("Loading ft_ou_model …")
        self._ft_ou_model = XGBClassifier()
        self._ft_ou_model.load_model(str(_FT_OU_MODEL_PATH))

        logger.info("Loading ht_ou_model …")
        self._ht_ou_model = XGBClassifier()
        self._ht_ou_model.load_model(str(_HT_OU_MODEL_PATH))

        # Load optional metadata (version info, accuracy stats)
        if _METADATA_PATH.exists():
            with open(_METADATA_PATH, encoding="utf-8") as f:
                self._metadata = json.load(f)
            logger.info(
                "Model metadata: version=%s trained_at=%s",
                self._metadata.get("version"),
                self._metadata.get("trained_at"),
            )
        else:
            logger.warning("metadata.json not found; version info will be unavailable.")

        self._loaded = True
        logger.info("All models loaded successfully.")

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
    def ht_ou_model(self) -> XGBClassifier:
        self._assert_loaded()
        return self._ht_ou_model  # type: ignore[return-value]

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

    def predict(self, feature_dict: dict) -> dict:
        """
        Run all three models for the given feature dictionary.

        Parameters
        ----------
        feature_dict:
            A flat ``{feature_name: value}`` dict.  Keys must cover at least
            every column listed in ``FEATURE_COLUMNS``.  Extra keys are
            silently ignored.

        Returns
        -------
        dict with keys:
            winner_pick     : 'Home' | 'Away'
            winner_trust    : float — confidence % (0–100)
            ft_ou_pick      : 'Over' | 'Under'
            ft_ou_trust     : float — confidence %
            ht_ou_pick      : 'Over' | 'Under'
            ht_ou_trust     : float — confidence %
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

        # ------------------------------------------------------------------
        # Build the input matrix in the exact column order required by models
        # ------------------------------------------------------------------
        try:
            row = [float(feature_dict[col]) for col in FEATURE_COLUMNS]
        except KeyError as exc:
            raise KeyError(
                f"Missing feature column in feature_dict: {exc}. "
                f"Required columns: {FEATURE_COLUMNS}"
            ) from exc

        # Shape: (1, n_features) — XGBoost expects a 2-D array.
        X = np.array([row], dtype=float)

        # ------------------------------------------------------------------
        # Winner model
        # ------------------------------------------------------------------
        winner_proba = self._loader.winner_model.predict_proba(X)[0]
        winner_class = int(np.argmax(winner_proba))
        winner_pick = "Home" if winner_class == 1 else "Away"
        # Trust = probability of the predicted class
        winner_trust = round(float(winner_proba[winner_class]) * 100, 1)

        # ------------------------------------------------------------------
        # Full-time O/U model
        # ------------------------------------------------------------------
        ft_ou_proba = self._loader.ft_ou_model.predict_proba(X)[0]
        ft_ou_class = int(np.argmax(ft_ou_proba))
        ft_ou_pick = "Over" if ft_ou_class == 1 else "Under"
        ft_ou_trust = round(float(ft_ou_proba[ft_ou_class]) * 100, 1)

        # ------------------------------------------------------------------
        # Halftime O/U model
        # ------------------------------------------------------------------
        ht_ou_proba = self._loader.ht_ou_model.predict_proba(X)[0]
        ht_ou_class = int(np.argmax(ht_ou_proba))
        ht_ou_pick = "Over" if ht_ou_class == 1 else "Under"
        ht_ou_trust = round(float(ht_ou_proba[ht_ou_class]) * 100, 1)

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
    Load all XGBoost models into the shared ``model_loader`` singleton.

    Call this from the FastAPI lifespan or startup event:

    .. code-block:: python

        from app.ml.predict import load_models

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            load_models()
            yield

    Logs a warning (rather than raising) if model files are absent so that
    the API can still start in environments where training has not yet run.
    """
    try:
        model_loader.load()
        logger.info("5Star Predictions models are ready.")
    except FileNotFoundError as exc:
        logger.warning(
            "⚠️  Could not load ML models: %s. "
            "Prediction endpoints will be unavailable until models are trained.",
            exc,
        )
