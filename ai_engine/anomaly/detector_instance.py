"""Singleton AnomalyDetector shared across all FastAPI requests."""
from anomaly.isolation_forest import AnomalyDetector

detector = AnomalyDetector(contamination=0.1, n_estimators=100)
detector.load()