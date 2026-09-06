from pydantic import BaseModel
from typing import List

class FeedbackPayload(BaseModel):
    tables: List[str]           # Tables that were in the query
    chosen_order: List[str]     # Join order the AI picked
    latency_ms: float           # Actual execution time measured by Java
    active_connections: int     # pg_stat_activity count at execution time