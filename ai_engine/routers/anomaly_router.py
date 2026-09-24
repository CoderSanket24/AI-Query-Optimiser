"""
anomaly_router.py
-----------------
GET /anomaly/status  -- Isolation Forest state and configuration.

Reflects the redesigned Part 5 v2:
  Forest now trained on [wait_time_ms, exec_time_ms, active_connections]
  to detect DoS patterns (high latency with low active connections).
  Applied AFTER execution in /feedback, not before in /optimize.
"""

import os
from fastapi import APIRouter
from anomaly.detector_instance import detector
from anomaly.isolation_forest  import FOREST_PATH, MIN_SAMPLES_TO_FIT, RETRAIN_EVERY_N

router = APIRouter(prefix="/anomaly", tags=["anomaly"])


@router.get("/status")
async def anomaly_status():
    size_kb = None
    if os.path.exists(FOREST_PATH):
        size_kb = round(os.stat(FOREST_PATH).st_size / 1024, 2)

    return {
        # Forest state
        "is_fitted":          detector.is_fitted,
        "n_samples_trained":  detector.n_samples_trained,
        "forest_file_exists": os.path.exists(FOREST_PATH),
        "forest_size_kb":     size_kb,

        # Configuration
        "contamination":      detector.contamination,
        "n_estimators":       detector.n_estimators,
        "min_samples_to_fit": MIN_SAMPLES_TO_FIT,
        "retrain_every_n":    RETRAIN_EVERY_N,

        # Design info
        "feature_vector":     ["wait_time_ms", "exec_time_ms", "active_connections"],
        "placement":          "post-execution in /feedback (not pre-query in /optimize)",
        "anomaly_action":     "reward=0, PPO training skipped (DoS protection)",
        "normal_reward":      "reward = -exec_time_ms / (1 + connections * 0.1)",
    }
