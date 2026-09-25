"""
analytics_store.py
------------------
Thread-safe in-memory store for query history.
Shared singleton used by queryOptimizer (record creation)
and feedback (record completion).
"""

import threading
from collections import deque
from datetime import datetime, timezone
from models.query_record import QueryRecord

_HISTORY_MAXLEN = 200

_store: deque = deque(maxlen=_HISTORY_MAXLEN)
_lock  = threading.Lock()
_counter = 0          # monotonic query ID


def new_record(
    tables:        list,
    original_sql:  str,
    chosen_order:  list,
    is_anomalous:  bool,
    anomaly_score: float,
) -> QueryRecord:
    """Create a pending record (outcome not yet known). Returns the record."""
    global _counter
    with _lock:
        _counter += 1
        qid = _counter
    rec = QueryRecord(
        query_id     = qid,
        timestamp    = datetime.now(timezone.utc).isoformat(),
        tables       = tables,
        original_sql = original_sql,
        chosen_order = chosen_order,
        is_anomalous = is_anomalous,
        anomaly_score= anomaly_score,
    )
    with _lock:
        _store.append(rec)
    return rec


def complete_record(
    query_id:          int,
    latency_ms:        float,
    active_connections:int,
    reward:            float,
    ppo_step:          int   = 0,
    ppo_loss:          float = 0.0,
    ppo_baseline:      float = 0.0,
    is_anomalous:      bool  = False,   # ← set True when DoS detected in /feedback
    anomaly_score:     float = 0.0,     # ← Isolation Forest score at feedback time
) -> None:
    """Fill in the outcome fields for an existing record."""
    with _lock:
        for rec in reversed(_store):
            if rec.query_id == query_id:
                rec.latency_ms          = latency_ms
                rec.active_connections  = active_connections
                rec.reward              = reward
                rec.ppo_step            = ppo_step
                rec.ppo_loss            = ppo_loss
                rec.ppo_baseline        = ppo_baseline
                rec.is_anomalous        = is_anomalous    # update from feedback
                rec.anomaly_score       = anomaly_score   # update from feedback
                return


def get_history(limit: int = 50, offset: int = 0) -> list:
    with _lock:
        items = list(_store)
    items.reverse()                        # newest first
    return [r.to_dict() for r in items[offset: offset + limit]]


def get_all() -> list:
    with _lock:
        return list(_store)


def total_count() -> int:
    with _lock:
        return len(_store)
