"""
isolation_forest.py  --  DoS-Aware Anomaly Detection (v3)
----------------------------------------------------------
PURPOSE:
  Detects Denial-of-Service (DoS) attacks by learning the normal
  relationship between (wait_time_ms, active_connections).

  exec_time_ms is deliberately EXCLUDED from the feature vector.
  Reason: exec_time can be high because of PPO's own bad join order
  (a legitimate learning signal). Including exec_time would cause the
  forest to flag those queries as DoS, block the reward, and prevent
  the PPO from ever learning to improve its join ordering.

FEATURE VECTOR (2D):
  [wait_time_ms, active_connections]

  wait_time_ms:       time waiting for a JDBC connection from HikariCP pool.
                      This is INDEPENDENT of join order — it reflects
                      DB congestion caused externally (e.g., DoS).
  active_connections: pg_stat_activity count at execution time.

DETECTION LOGIC:
  Normal patterns:
    Low wait  + any connections   -> Idle or busy DB, queries going through  OK
    High wait + High connections  -> DB under legitimate heavy load           OK

  Anomaly pattern (DoS signal):
    High wait + LOW connections   -> Victims waiting behind a DoS query      ALERT
    (hacker's cartesian product holds the DB but few total connections)

  The hacker's OWN first query is NOT directly detectable via wait_time
  (it enters with wait=0). Detection happens from the VICTIM side once
  subsequent legitimate queries start experiencing high wait + low connections.
  The AST Firewall handles the most obvious cartesian-product patterns up front.

PLACEMENT:
  Called AFTER query execution inside /feedback (not before in /optimize).
  Anomaly -> reward = 0, PPO training skipped (protects RL from poisoned signal).
  Normal  -> reward computed from exec_time_ms, PPO trains normally.

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

    Feature vector per execution (2D):
      [wait_time_ms, active_connections]

    exec_time_ms is intentionally excluded — it can be high due to
    PPO's own suboptimal join order, not because of a DoS attack.

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
        Train on list of [wait_time_ms, active_connections] vectors.
        Automatically saves to disk after fitting.
        """
        X = np.array(samples, dtype=np.float32)
        self._forest = SKLearnIF(
            n_estimators  = self.n_estimators,
            contamination = self.contamination,
            random_state  = 42,
        )
        self._forest.fit(X)
        self.is_fitted         = True
        self.n_samples_trained = len(samples)
        print(
            f"[Forest] Fitted on {self.n_samples_trained} samples "
            f"| contamination={self.contamination} "
            f"| features=[wait_ms, connections]  "
            f"(exec_ms excluded -- not a DoS signal)"
        )
        self.save()

    # ------------------------------------------------------------------
    def predict(self,
                wait_time_ms:       float,
                active_connections: int) -> tuple:
        """
        Predict whether this execution reflects a DoS pattern.

        Args:
            wait_time_ms:       HikariCP pool wait time in ms
            active_connections: pg_stat_activity count

        Returns:
            (is_anomaly: bool, score: float)
            score < 0 = anomalous (isolated quickly by forest)
            score > 0 = normal    (hard to isolate)

        If not yet fitted -> returns (False, 0.0) so PPO trains normally
        until we have enough samples for a reliable boundary.
        """
        if not self.is_fitted:
            return False, 0.0

        x     = np.array([[wait_time_ms, float(active_connections)]],
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
        print(f"[Forest] Saved -> {FOREST_PATH}")

    def load(self) -> bool:
        if not os.path.exists(FOREST_PATH):
            print(f"[Forest] No saved model. Will fit after {MIN_SAMPLES_TO_FIT} samples.")
            return False
        data                   = joblib.load(FOREST_PATH)
        self._forest           = data["forest"]
        self.n_samples_trained = data["n_samples"]
        self.is_fitted         = True
        print(
            f"[Forest] Loaded: trained on {self.n_samples_trained} samples "
            f"| features=[wait_ms, connections]"
        )
        return True

    # ------------------------------------------------------------------
    @property
    def forest_path(self) -> str:
        return FOREST_PATH

    @property
    def forest_exists(self) -> bool:
        return os.path.exists(FOREST_PATH)