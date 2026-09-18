from pydantic import BaseModel
from typing import List, Optional


class FeedbackPayload(BaseModel):
    tables:             List[str]
    chosen_order:       List[str]
    latency_ms:         float
    active_connections: int
    query_id:           Optional[int]   = None   # links to analytics record
    log_prob_old:       Optional[float] = None   # log π_old for PPO ratio (Part 7)