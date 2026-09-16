import os
from fastapi import APIRouter
from anomaly.detector_instance import detector
from anomaly.isolation_forest import FOREST_PATH, MIN_SAMPLES_TO_FIT, RETRAIN_EVERY_N

router = APIRouter(prefix="/anomaly", tags=["anomaly"])


@router.get("/status")
async def anomaly_status():
    """Returns the current state of the Isolation Forest anomaly detector."""
    info = {
        "is_fitted":          detector.is_fitted,
        "n_samples_trained":  detector.n_samples_trained,
        "contamination":      detector.contamination,
        "min_samples_to_fit": MIN_SAMPLES_TO_FIT,
        "retrain_every_n":    RETRAIN_EVERY_N,
        "forest_file_exists": detector.forest_exists,
        "forest_path":        FOREST_PATH,
    }
    if detector.forest_exists:
        info["forest_size_kb"] = round(os.stat(FOREST_PATH).st_size / 1024, 2)
    return info
