from pydantic import BaseModel
from typing import List, Optional


class FeedbackPayload(BaseModel):
    tables:             List[str]
    chosen_order:       List[str]

    # Timing (split by QueryTelemetryService.java)
    latency_ms:         float           # total = wait + exec (kept for compatibility)
    exec_time_ms:       float = 0.0     # actual PostgreSQL execution → PPO reward signal
    wait_time_ms:       float = 0.0     # HikariCP pool wait       → Isolation Forest signal

    active_connections: int

    # Analytics + PPO linkage
    query_id:           Optional[int]   = None
    log_prob_old:       Optional[float] = None