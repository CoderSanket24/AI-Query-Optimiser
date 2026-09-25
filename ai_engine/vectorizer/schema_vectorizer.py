"""
schema_vectorizer.py
--------------------
Converts a list of SQL table names into a real [N_tables x 11] PyTorch tensor
by querying live schema statistics from PostgreSQL.

Feature vector per table (11 dimensions):
  [0]  log_row_count        - Estimated row count (log-normalised)
  [1]  log_page_count       - Number of 8KB disk pages (log-normalised)
  [2]  log_col_count        - Number of columns (log-normalised)
  [3]  log_index_count      - Number of indexes (log-normalised)
  [4]  has_primary_key      - Binary: 1 if PK exists
  [5]  has_foreign_key      - Binary: 1 if any FK exists
  [6]  log_table_size_bytes - Physical table size (log-normalised)
  [7]  avg_row_width_norm   - Avg bytes per row (log-normalised)
  [8]  row_density          - Rows per page, normalised to [0,1]
  [9]  is_in_query          - Always 1.0 (table is referenced in this query)
  [10] row_growth_rate      - Normalised row growth rate since last fetch
                              0.0 = stable / shrinking
                              0.5 = moderate growth
                              1.0 = very fast growth (≥ MAX_GROWTH_RATE rows/sec)

                              Dynamic signal: PPO sees not just table size but
                              HOW FAST the table is growing right now.
                              Enables true dynamic production DB adaptation.

Caching:
  Stats are cached per table with TTL = 60 seconds (reduced from 300s
  to capture growth events faster in production-like workloads).
  Growth rate is computed from the PREVIOUS cache entry vs the new fetch.
"""

import math
import time
import psycopg2
import torch

# DB connection config (matches application.properties)
_DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "dbname":   "ai_optimizer",
    "user":     "postgres",
    "password": "password",
}

# TTL cache: key = table_name, value = (timestamp, stats_dict)
_CACHE = {}
_CACHE_TTL_SECONDS = 60        # reduced from 300s — faster adaptation

# Upper bounds for log-normalisation (just above largest IMDB table values)
_MAX_ROWS        = 4e7         # cast_info ~ 36M rows
_MAX_PAGES       = 3e5         # cast_info ~ 252K pages
_MAX_COLS        = 20          # title has 12 cols
_MAX_INDEXES     = 10
_MAX_SIZE_BYTES  = 2.5e9       # ~2.5 GB
_MAX_ROW_WIDTH   = 2000        # bytes per row
_MAX_ROW_DENS    = 500         # rows per page

# Growth rate: rows-per-second that maps to normalised value 1.0
# ~100K rows/sec is extremely fast growth for a production OLTP table
_MAX_GROWTH_RATE = 1e5         # rows per second


def _log_norm(value, max_val):
    """Log-normalise value into [0, 1]. Uses log1p so 0 maps to 0."""
    if value <= 0 or max_val <= 0:
        return 0.0
    return min(math.log1p(value) / math.log1p(max_val), 1.0)


def _fetch_stats(table_name):
    """Query PostgreSQL for schema statistics for a single table."""
    conn = psycopg2.connect(**_DB_CONFIG)
    try:
        cur = conn.cursor()

        # 1. Row count and page count from pg_class
        cur.execute("""
            SELECT reltuples, relpages
            FROM   pg_class
            WHERE  relname = %s AND relkind = 'r';
        """, (table_name,))
        row = cur.fetchone()
        row_count  = float(row[0]) if row and row[0] > 0 else 1.0
        page_count = float(row[1]) if row and row[1] > 0 else 1.0

        # 2. Column count from information_schema
        cur.execute("""
            SELECT COUNT(*)
            FROM   information_schema.columns
            WHERE  table_schema = 'public' AND table_name = %s;
        """, (table_name,))
        col_count = float(cur.fetchone()[0] or 1)

        # 3. Index count
        cur.execute("""
            SELECT COUNT(*)
            FROM   pg_indexes
            WHERE  schemaname = 'public' AND tablename = %s;
        """, (table_name,))
        index_count = float(cur.fetchone()[0] or 0)

        # 4. Primary key presence
        cur.execute("""
            SELECT COUNT(*)
            FROM   information_schema.table_constraints
            WHERE  table_schema = 'public'
              AND  table_name = %s
              AND  constraint_type = 'PRIMARY KEY';
        """, (table_name,))
        has_pk = 1.0 if cur.fetchone()[0] > 0 else 0.0

        # 5. Foreign key presence
        cur.execute("""
            SELECT COUNT(*)
            FROM   information_schema.table_constraints
            WHERE  table_schema = 'public'
              AND  table_name = %s
              AND  constraint_type = 'FOREIGN KEY';
        """, (table_name,))
        has_fk = 1.0 if cur.fetchone()[0] > 0 else 0.0

        # 6. Physical size in bytes
        cur.execute("SELECT pg_relation_size(%s);", (table_name,))
        size_bytes = float(cur.fetchone()[0] or 1)

        return {
            "row_count":   row_count,
            "page_count":  page_count,
            "col_count":   col_count,
            "index_count": index_count,
            "has_pk":      has_pk,
            "has_fk":      has_fk,
            "size_bytes":  size_bytes,
        }

    except Exception as e:
        print(f"[Vectorizer] DB error for table '{table_name}': {e}")
        # Neutral mid-range fallback so the model still gets valid input
        return {
            "row_count":   1000.0,
            "page_count":  10.0,
            "col_count":   5.0,
            "index_count": 1.0,
            "has_pk":      1.0,
            "has_fk":      0.0,
            "size_bytes":  81920.0,
        }
    finally:
        conn.close()


def _get_stats_cached(table_name):
    """
    Return fresh stats from PostgreSQL, caching for TTL seconds.

    Also computes row_growth_rate by comparing the new row_count with
    the PREVIOUS cached row_count and the elapsed time between fetches.
    This is the key feature that makes the vectorizer truly dynamic:
    the PPO sees how fast each table is currently growing.
    """
    now = time.time()

    if table_name in _CACHE:
        cached_time, cached_stats = _CACHE[table_name]

        # Still fresh — return cached stats (growth_rate already stored)
        if now - cached_time < _CACHE_TTL_SECONDS:
            return cached_stats

        # Cache expired — fetch new stats and compute growth rate
        prev_row_count = cached_stats["row_count"]
        prev_time      = cached_time
    else:
        prev_row_count = None
        prev_time      = None

    # Fetch fresh stats from PostgreSQL
    stats = _fetch_stats(table_name)

    # ── Compute row growth rate ─────────────────────────────────────
    if prev_row_count is not None and prev_time is not None:
        elapsed_seconds  = max(now - prev_time, 1.0)   # avoid div-by-zero
        row_delta        = stats["row_count"] - prev_row_count
        rows_per_second  = max(row_delta, 0.0) / elapsed_seconds  # only positive growth
        growth_rate_norm = min(rows_per_second / _MAX_GROWTH_RATE, 1.0)
    else:
        # First time we've seen this table — no growth info yet
        growth_rate_norm = 0.0

    stats["growth_rate_norm"] = growth_rate_norm

    # Store in cache
    _CACHE[table_name] = (now, stats)

    print(
        f"[Vectorizer] Fetched stats for '{table_name}': "
        f"rows={int(stats['row_count']):,}  pages={int(stats['page_count']):,}  "
        f"cols={int(stats['col_count'])}  indexes={int(stats['index_count'])}  "
        f"pk={int(stats['has_pk'])}  fk={int(stats['has_fk'])}  "
        f"size={stats['size_bytes'] / 1e6:.1f}MB  "
        f"growth_rate={growth_rate_norm:.4f}"
    )
    return stats


def _build_feature_vector(table_name):
    """Return an 11-dimensional feature vector for one table."""
    s = _get_stats_cached(table_name)

    avg_row_width = s["size_bytes"] / max(s["row_count"], 1)
    row_density   = s["row_count"]  / max(s["page_count"], 1)

    return [
        _log_norm(s["row_count"],   _MAX_ROWS),        # [0]  row count
        _log_norm(s["page_count"],  _MAX_PAGES),        # [1]  page count
        _log_norm(s["col_count"],   _MAX_COLS),         # [2]  column count
        _log_norm(s["index_count"], _MAX_INDEXES),      # [3]  index count
        s["has_pk"],                                    # [4]  has primary key
        s["has_fk"],                                    # [5]  has foreign key
        _log_norm(s["size_bytes"],  _MAX_SIZE_BYTES),   # [6]  table size bytes
        _log_norm(avg_row_width,    _MAX_ROW_WIDTH),    # [7]  avg row width
        min(row_density / _MAX_ROW_DENS, 1.0),         # [8]  row density
        1.0,                                            # [9]  is in query
        s["growth_rate_norm"],                          # [10] row growth rate (NEW)
    ]


def build_state_tensor(tables):
    """
    Public API called by queryOptimizer.py.
    Returns a float32 tensor of shape [len(tables), 11].
    """
    vectors = [_build_feature_vector(t) for t in tables]
    tensor  = torch.tensor(vectors, dtype=torch.float32)
    print(f"[Vectorizer] Built state tensor: shape={list(tensor.shape)}")
    return tensor