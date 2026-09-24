"""
isolation_forest.py  --  DoS-Aware Anomaly Detection
------------------------------------------------------
Redesigned (Part 5 v2):

PURPOSE:
  Detects Denial-of-Service (DoS) / bad-query attacks by learning the
  normal relationship between (wait_time_ms, exec_time_ms, active_connections).

  Normal patterns:
    High wait + High connections  → DB under legitimate load      ✅
    Low wait  + Low exec          → Idle DB, fast query            ✅

  Anomaly patterns (DoS signals):
    High exec  + Low connections  → Hacker's own slow query        ❌
    High wait  + Low connections  → Victims waiting behind DoS     ❌

PLACEMENT:
  Called AFTER query execution inside /feedback (not before in /optimize).
  Anomaly → reward = 0, PPO training skipped (protects RL from bad signal).
  Normal  → reward computed from exec_time_ms, PPO trains normally.

FEATURE VECTOR (3D):
  [wait_time_ms, exec_time_ms, active_connections]

  wait_time_ms:       time waiting for a JDBC connection from HikariCP pool
  exec_time_ms:       actual PostgreSQL query execution time
  active_connections: pg_stat_activity count at execution time

Thresholds:
  MIN_SAMPLES_TO_FIT = 10   first fit after 10 feedback samples
  RETRAIN_EVERY_N    = 5    refit every 5 new samples thereafter
  contamination      = 0.05 expect 5% of executions to be anomalous
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest as SKLearnIF

_BASE_DIR          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOREST_PATH        = os.path.join(_BASE_DIR, "checkpoints", "isolation_forest.pkl")
MIN_SAMPLES_TO_FIT = 10
RETRAIN_EVERY_N    = 5


class AnomalyDetector:
    """
    Sklearn IsolationForest wrapper with joblib persistence.

    Feature vector per execution:
      [wait_time_ms, exec_time_ms, active_connections]

    Lifecycle:
      1. Created at startup in detector_instance.py
      2. load()    -- resumes saved model if pkl exists
      3. fit()     -- called from /feedback once enough samples accumulated
      4. predict() -- called from /feedback after each execution
                      returns (is_anomaly: bool, score: float)
    """

    def __init__(self, contamination: float = 0.05, n_estimators: int = 100):
        self.contamination     = contamination
        self.n_estimators      = n_estimators
        self._forest           = None
        self.is_fitted         = False
        self.n_samples_trained = 0

    # ------------------------------------------------------------------
    def fit(self, samples: list) -> None:
        """
        Train on list of [wait_time_ms, exec_time_ms, active_connections] vectors.
        Automatically saves to disk after fitting.
        """
        X = np.array(samples, dtype=np.float32)
        self._forest = SKLearnIF(
            n_estimators   = self.n_estimators,
            contamination  = self.contamination,
            random_state   = 42,
        )
        self._forest.fit(X)
        self.is_fitted         = True
        self.n_samples_trained = len(samples)
        print(f"[Forest] Fitted on {self.n_samples_trained} samples "
              f"| contamination={self.contamination} "
              f"| features=[wait_ms, exec_ms, connections]")
        self.save()

    # ------------------------------------------------------------------
    def predict(self, wait_time_ms: float,
                      exec_time_ms: float,
                      active_connections: int) -> tuple:
        """
        Predict whether this execution is anomalous.

        Returns:
          (is_anomaly: bool, score: float)
          score < 0 = anomalous (isolated quickly)
          score > 0 = normal    (hard to isolate)

        If not yet fitted → returns (False, 0.0) so training proceeds normally
        until we have enough samples.
        """
        if not self.is_fitted:
            return False, 0.0

        x     = np.array([[wait_time_ms, exec_time_ms, active_connections]],
                          dtype=np.float32)
        score = float(self._forest.decision_function(x)[0])
        label = int(self._forest.predict(x)[0])
        return (label == -1), round(score, 4)

    # ------------------------------------------------------------------
    def save(self) -> None:
        os.makedirs(os.path.dirname(FOREST_PATH), exist_ok=True)
        joblib.dump({
            "forest":   self._forest,
            "n_samples": self.n_samples_trained,
        }, FOREST_PATH)
        print(f"[Forest] Saved → {FOREST_PATH}")

    def load(self) -> bool:
        if not os.path.exists(FOREST_PATH):
            print(f"[Forest] No saved model. Will fit after {MIN_SAMPLES_TO_FIT} samples.")
            return False
        data                   = joblib.load(FOREST_PATH)
        self._forest           = data["forest"]
        self.n_samples_trained = data["n_samples"]
        self.is_fitted         = True
        print(f"[Forest] Loaded: trained on {self.n_samples_trained} samples "
              f"| features=[wait_ms, exec_ms, connections]")
        return True

    # ------------------------------------------------------------------
    @property
    def forest_path(self) -> str:
        return FOREST_PATH

    @property
    def forest_exists(self) -> bool:
        return os.path.exists(FOREST_PATH)