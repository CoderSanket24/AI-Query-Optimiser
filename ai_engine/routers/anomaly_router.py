"""
anomaly_router.py
-----------------
GET /anomaly/status  -- Isolation Forest state and configuration.

Reflects the redesigned Part 5 v3:
  Forest trained on [wait_time_ms, active_connections]  (2D, exec_time EXCLUDED)
  exec_time excluded because it can be high due to PPO's own bad join order --
  including it would falsely flag legitimate PPO learning as DoS.
  Detection works via victim pattern: high wait + low connections.
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

        # Design info (v3)
        "feature_vector":     ["wait_time_ms", "active_connections"],
        "exec_time_excluded": "yes -- exec can be high due to PPO bad join order, not DoS",
        "detection_pattern":  "high wait_time + LOW connections = DoS victim pattern",
        "placement":          "post-execution in /feedback (not pre-query in /optimize)",
        "anomaly_action":     "reward=0, PPO training skipped (DoS protection)",
        "normal_reward":      "reward = -exec_time_ms / (1 + connections * 0.1)",
    }
