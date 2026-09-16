"""
isolation_forest.py  --  Layer 2 Anomaly Detection
---------------------------------------------------
Each query is mean-pooled into a 10-dim vector (from the schema vectorizer).
IsolationForest learns the distribution of normal queries and flags outliers.
Anomalous queries skip the pg_hint_plan hint and run as plain SQL for safety.

Thresholds:
  MIN_SAMPLES_TO_FIT = 5    first fit after 5 buffered experiences
  RETRAIN_EVERY_N    = 3    refit every 3 new experiences thereafter
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest as SKLearnIF

_BASE_DIR          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOREST_PATH        = os.path.join(_BASE_DIR, "checkpoints", "isolation_forest.pkl")
MIN_SAMPLES_TO_FIT = 5
RETRAIN_EVERY_N    = 3


class AnomalyDetector:
    """
    Sklearn IsolationForest wrapper with joblib persistence.

    Lifecycle:
      1. Created at startup in detector_instance.py
      2. load()    -- resumes saved model if available
      3. predict() -- called on every /optimize request (no-op if not fitted)
      4. fit()     -- called from /feedback once enough samples are buffered
    """

    def __init__(self, contamination: float = 0.1, n_estimators: int = 100):
        self.contamination     = contamination
        self.n_estimators      = n_estimators
        self._forest           = None
        self.is_fitted         = False
        self.n_samples_trained = 0

    def fit(self, feature_vectors: list) -> None:
        X = np.array(feature_vectors, dtype=np.float32)
        self._forest = SKLearnIF(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=42,
        )
        self._forest.fit(X)
        self.is_fitted         = True
        self.n_samples_trained = len(feature_vectors)
        print(f"[Forest] Fitted on {self.n_samples_trained} samples | contamination={self.contamination}")
        self.save()

    def predict(self, feature_vector: list) -> tuple:
        """Returns (is_anomaly: bool, anomaly_score: float). Score < 0 = anomalous."""
        if not self.is_fitted:
            return False, 0.0
        x     = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
        score = float(self._forest.decision_function(x)[0])
        label = int(self._forest.predict(x)[0])
        return (label == -1), round(score, 4)

    def save(self) -> None:
        os.makedirs(os.path.dirname(FOREST_PATH), exist_ok=True)
        joblib.dump({"forest": self._forest, "n_samples": self.n_samples_trained}, FOREST_PATH)
        print(f"[Forest] Saved -> {FOREST_PATH}")

    def load(self) -> bool:
        if not os.path.exists(FOREST_PATH):
            print(f"[Forest] No saved model. Will fit after {MIN_SAMPLES_TO_FIT} samples.")
            return False
        data                   = joblib.load(FOREST_PATH)
        self._forest           = data["forest"]
        self.n_samples_trained = data["n_samples"]
        self.is_fitted         = True
        print(f"[Forest] Loaded: trained on {self.n_samples_trained} samples")
        return True

    @property
    def forest_path(self) -> str:
        return FOREST_PATH

    @property
    def forest_exists(self) -> bool:
        return os.path.exists(FOREST_PATH)


def get_query_vector(state_tensor) -> list:
    """Mean-pool [N_tables, 10] -> [10] fixed-size query representation."""
    return state_tensor.mean(dim=0).detach().tolist()