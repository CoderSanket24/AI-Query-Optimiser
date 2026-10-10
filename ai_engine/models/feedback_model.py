from pydantic import BaseModel
from typing import List, Optional


class FeedbackPayload(BaseModel):
    tables:             List[str]
    real_names:         Optional[List[str]] = None  # physical names for schema stats
    chosen_order:       List[str]

    # Timing (split by QueryTelemetryService.java using System.nanoTime())
    latency_ms:         float           # total = wait + exec (kept for compatibility)
    exec_time_ms:       float = -1.0    # PostgreSQL execution time → PPO reward signal
                                        # -1.0 = EXEC_MISSING sentinel (nanoTime failed)
    wait_time_ms:       float = 0.0     # HikariCP pool wait       → Isolation Forest signal

    active_connections: int

    # Analytics linkage — query_id only; log_prob_old is read from server-side
    # pending store (experience_buffer) and must NOT be accepted from the client.
    query_id:           Optional[int]   = None
    # log_prob_old intentionally removed from the schema to prevent client tampering.