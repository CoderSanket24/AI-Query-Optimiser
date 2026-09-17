from pydantic import BaseModel
from typing import List, Optional


class FeedbackPayload(BaseModel):
    tables:             List[str]
    chosen_order:       List[str]
    latency_ms:         float
    active_connections: int
    query_id:           Optional[int] = None    # links feedback to analytics record