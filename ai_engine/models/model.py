from pydantic import BaseModel
from typing import List, Optional


class QueryState(BaseModel):
    tables:          List[str]            # aliases — identifiers used in pg_hint_plan hints
    real_names:      Optional[List[str]] = None  # physical table names for schema stats
                                                  # falls back to `tables` when absent
    join_conditions: List[str]            # join-graph edges: each entry is "leftAlias,rightAlias"
                                          # extracted from ON clauses by Java AST walker.
                                          # Empty list = single-table query or no parsed edges.
    original_sql:    str

    # Security flag set by Java after alias whitelist check.
    # False means at least one alias contains characters that could escape a
    # /* ... */ comment (e.g. */ or ;).  When False, Python must NOT emit a hint.
    aliases_safe:    bool = True