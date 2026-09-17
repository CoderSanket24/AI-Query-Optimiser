"""
query_record.py
---------------
In-memory record of a single query decision + outcome.
Appended to the analytics history after every feedback call.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List


@dataclass
class QueryRecord:
    # Identity
    query_id:     int
    timestamp:    str                   # ISO-8601

    # Input
    tables:       List[str]
    original_sql: str

    # Decision
    chosen_order:  List[str]
    is_anomalous:  bool
    anomaly_score: float

    # Outcome (filled after feedback)
    latency_ms:         float = 0.0
    active_connections: int   = 0
    reward:             float = 0.0

    # Training
    ppo_step:    int   = 0
    ppo_loss:    float = 0.0
    ppo_baseline:float = 0.0

    def to_dict(self) -> dict:
        return {
            "query_id":          self.query_id,
            "timestamp":         self.timestamp,
            "tables":            self.tables,
            "chosen_order":      self.chosen_order,
            "is_anomalous":      self.is_anomalous,
            "anomaly_score":     self.anomaly_score,
            "latency_ms":        self.latency_ms,
            "active_connections":self.active_connections,
            "reward":            self.reward,
            "ppo_step":          self.ppo_step,
            "ppo_loss":          self.ppo_loss,
            "ppo_baseline":      self.ppo_baseline,
        }
