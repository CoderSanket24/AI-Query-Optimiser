from pydantic import BaseModel
from typing import List, Optional


class QueryState(BaseModel):
    tables:          List[str]           # aliases — identifiers used in pg_hint_plan hints
    real_names:      Optional[List[str]] = None  # physical table names for schema stats
                                                  # falls back to `tables` when absent
    join_conditions: List[str]
    original_sql:    str