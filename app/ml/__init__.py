# app/ml/__init__.py
#
# Re-export the two ML sub-modules under the names that the rest of the
# codebase expects:
#   - `feature_engineering`  →  app.ml.features
#   - `prediction_engine`    →  app.ml.predict
#
# This lets `from app.ml import feature_engineering, prediction_engine` work
# without changing any other file in the project.

from app.ml import features as feature_engineering  # noqa: F401
from app.ml import predict as prediction_engine      # noqa: F401
