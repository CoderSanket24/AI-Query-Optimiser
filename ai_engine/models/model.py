from pydantic import BaseModel
from typing import List

class QueryState(BaseModel):
    tables: List[str]
    join_conditions: List[str]
    original_sql: str