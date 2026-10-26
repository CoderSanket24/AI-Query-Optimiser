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

Batching:
  All cache-missing tables of ONE request are fetched with a SINGLE
  connection and ONE query per statistic (each = ANY(%s) / unnest), instead
  of 6 queries per table on its own connection.  This lowers cold-cache
  latency without changing feature definitions, normalisation constants,
  TTL, or growth-rate semantics.

Thread safety:
  build_state_tensor() is also called from the feedback worker thread
  (train_step REINFORCE path).  The _CACHE read-modify-write (growth
  computation + store) is guarded by _cache_lock; the DB fetch runs
  OUTSIDE that lock.
"""

import math
import os
import time
import threading
import psycopg2
import torch

# DB connection config — overridable via environment variables (defaults match
# application.properties for local development).
_DB_CONFIG = {
    "host":     os.environ.get("DB_HOST",     "localhost"),
    "port":     int(os.environ.get("DB_PORT", "5432")),
    "dbname":   os.environ.get("DB_NAME",     "ai_optimizer"),
    "user":     os.environ.get("DB_USERNAME", "postgres"),
    "password": os.environ.get("DB_PASSWORD", "password"),
}

# TTL cache: key = table_name, value = (timestamp, stats_dict)
_CACHE = {}
_CACHE_TTL_SECONDS = 60        # reduced from 300s — faster adaptation
_cache_lock = threading.Lock()

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

# Neutral mid-range fallback so the model still gets valid input when the DB
# is unreachable OR a specific relation does not exist (legacy raise path).
_NEUTRAL_STATS = {
    "row_count":   1000.0,
    "page_count":  10.0,
    "col_count":   5.0,
    "index_count": 1.0,
    "has_pk":      1.0,
    "has_fk":      0.0,
    "size_bytes":  81920.0,
}


def _log_norm(value, max_val):
    """Log-normalise value into [0, 1]. Uses log1p so 0 maps to 0."""
    if value <= 0 or max_val <= 0:
        return 0.0
    return min(math.log1p(value) / math.log1p(max_val), 1.0)


def _compute_growth(prev_row_count, prev_time, new_row_count, now):
    """
    Identical to the legacy growth computation: compare the new row_count
    against the PREVIOUS cached row_count over the elapsed time, capped to 1.0.
    """
    if prev_row_count is not None and prev_time is not None:
        elapsed_seconds  = max(now - prev_time, 1.0)   # avoid div-by-zero
        row_delta        = new_row_count - prev_row_count
        rows_per_second  = max(row_delta, 0.0) / elapsed_seconds  # only positive growth
        return min(rows_per_second / _MAX_GROWTH_RATE, 1.0)
    return 0.0


def _fetch_stats_batch(table_names):
    """
    Fetch [row_count, page_count, col_count, index_count, has_pk, has_fk,
    size_bytes] for a list of tables using ONE connection and one query per
    statistic (batched with = ANY(%s) / unnest(%s::text[])).

    Values are identical to the legacy per-table fetch, including its
    missing-table behaviour: pg_relation_size(to_regclass(n)) resolves via
    search_path exactly like the legacy text cast, and a NULL result (relation
    not found — which made the legacy text cast RAISE) falls back to the
    neutral dict for THAT table only.
    """
    names = list(table_names)
    conn = psycopg2.connect(**_DB_CONFIG)   # OUTSIDE try (same as legacy)

    _stage = "cursor"
    try:
        cur = conn.cursor()

        # 1. Row + page counts from pg_class.  No schema filter (same as
        #    legacy); the FIRST occurrence per relname wins, matching
        #    legacy fetchone() scan order.
        _stage = "pg_class"
        cur.execute(
            "SELECT relname, reltuples, relpages "
            "FROM   pg_class "
            "WHERE  relname = ANY(%s) AND relkind = 'r';",
            (names,),
        )
        pg_first = {}
        for relname, reltuples, relpages in cur.fetchall():
            if relname not in pg_first:
                pg_first[relname] = (reltuples, relpages)

        # 2. Column count (information_schema.columns)
        _stage = "columns"
        cur.execute(
            "SELECT table_name, COUNT(*) "
            "FROM   information_schema.columns "
            "WHERE  table_schema = 'public' AND table_name = ANY(%s) "
            "GROUP BY table_name;",
            (names,),
        )
        col_counts = dict(cur.fetchall())

        # 3. Index count (pg_indexes)
        _stage = "indexes"
        cur.execute(
            "SELECT tablename, COUNT(*) "
            "FROM   pg_indexes "
            "WHERE  schemaname = 'public' AND tablename = ANY(%s) "
            "GROUP BY tablename;",
            (names,),
        )
        index_counts = dict(cur.fetchall())

        # 4. Primary key presence (information_schema.table_constraints)
        _stage = "pk"
        cur.execute(
            "SELECT table_name, COUNT(*) "
            "FROM   information_schema.table_constraints "
            "WHERE  table_schema = 'public' AND table_name = ANY(%s) "
            "  AND  constraint_type = 'PRIMARY KEY' "
            "GROUP BY table_name;",
            (names,),
        )
        pk_counts = dict(cur.fetchall())

        # 5. Foreign key presence (information_schema.table_constraints)
        _stage = "fk"
        cur.execute(
            "SELECT table_name, COUNT(*) "
            "FROM   information_schema.table_constraints "
            "WHERE  table_schema = 'public' AND table_name = ANY(%s) "
            "  AND  constraint_type = 'FOREIGN KEY' "
            "GROUP BY table_name;",
            (names,),
        )
        fk_counts = dict(cur.fetchall())

        # 6. Physical size.  search_path resolution via to_regclass() — same
        #    as the legacy text cast.  NULL => relation not found.
        _stage = "size"
        cur.execute(
            "SELECT n, pg_relation_size(to_regclass(n)) "
            "FROM   unnest(%s::text[]) AS n;",
            (names,),
        )
        sizes = dict(cur.fetchall())

    except Exception as exc:
        # Log which batched query failed, then fall back for the whole set
        # (legacy isolated per table; batching couples them — reported).
        print(f"[Vectorizer] DB error (batch, stage='{_stage}', "
              f"tables={len(names)}): {exc}")
        return {t: dict(_NEUTRAL_STATS) for t in names}
    finally:
        conn.close()

    result = {}
    for t in names:
        if t in sizes and sizes[t] is None:
            # relation does not exist -> legacy text cast raised -> neutral
            # fallback for THIS table only.
            result[t] = dict(_NEUTRAL_STATS)
            continue

        reltuples, relpages = pg_first.get(t, (None, None))
        row_count   = float(reltuples) if (reltuples is not None and reltuples > 0) else 1.0
        page_count  = float(relpages)  if (relpages  is not None and relpages  > 0) else 1.0
        col_count   = float(col_counts.get(t, 0) or 1)
        index_count = float(index_counts.get(t, 0) or 0)
        has_pk = 1.0 if pk_counts.get(t, 0) > 0 else 0.0
        has_fk = 1.0 if fk_counts.get(t, 0) > 0 else 0.0
        size_bytes  = float(sizes.get(t) or 1)

        result[t] = {
            "row_count":   row_count,
            "page_count":  page_count,
            "col_count":   col_count,
            "index_count": index_count,
            "has_pk":      has_pk,
            "has_fk":      has_fk,
            "size_bytes":  size_bytes,
        }
    return result


def _build_feature_vector(stats):
    """Return an 11-dimensional feature vector from a stats dict."""
    avg_row_width = stats["size_bytes"] / max(stats["row_count"], 1)
    row_density   = stats["row_count"]  / max(stats["page_count"], 1)

    return [
        _log_norm(stats["row_count"],   _MAX_ROWS),        # [0]  row count
        _log_norm(stats["page_count"],  _MAX_PAGES),        # [1]  page count
        _log_norm(stats["col_count"],   _MAX_COLS),         # [2]  column count
        _log_norm(stats["index_count"], _MAX_INDEXES),      # [3]  index count
        stats["has_pk"],                                    # [4]  has primary key
        stats["has_fk"],                                    # [5]  has foreign key
        _log_norm(stats["size_bytes"],  _MAX_SIZE_BYTES),   # [6]  table size bytes
        _log_norm(avg_row_width,        _MAX_ROW_WIDTH),    # [7]  avg row width
        min(row_density / _MAX_ROW_DENS, 1.0),             # [8]  row density
        1.0,                                               # [9]  is in query
        stats["growth_rate_norm"],                         # [10] row growth rate
    ]


def _get_stats_batched(table_names):
    """
    Return {table_name: stats_dict} using the cache, batch-fetching misses.

    Fresh cache hits are returned as-is; stale/missing tables are fetched in
    ONE batch, then growth is computed per table vs its previous cached
    row_count and the entry is re-cached.  The _CACHE read-modify-write is
    guarded by _cache_lock; the DB fetch runs OUTSIDE that lock.
    """
    now = time.time()
    result = {}
    missing = []
    prev_by_table = {}

    with _cache_lock:
        for t in table_names:
            if t in _CACHE:
                cached_time, cached_stats = _CACHE[t]
                if now - cached_time < _CACHE_TTL_SECONDS:
                    result[t] = cached_stats
                    continue
                prev_by_table[t] = (cached_stats["row_count"], cached_time)
            else:
                prev_by_table[t] = (None, None)
            missing.append(t)

    if missing:
        fetched = _fetch_stats_batch(missing)   # DB I/O OUTSIDE the lock

        to_print = []
        with _cache_lock:
            for t in missing:
                stats = fetched[t]
                growth = _compute_growth(
                    prev_by_table[t][0], prev_by_table[t][1],
                    stats["row_count"], now)
                stats["growth_rate_norm"] = growth
                _CACHE[t] = (now, stats)
                result[t] = stats
                to_print.append((t, stats, growth))

        for t, stats, growth in to_print:
            print(
                f"[Vectorizer] Fetched stats for '{t}': "
                f"rows={int(stats['row_count']):,}  "
                f"pages={int(stats['page_count']):,}  "
                f"cols={int(stats['col_count'])}  "
                f"indexes={int(stats['index_count'])}  "
                f"pk={int(stats['has_pk'])}  fk={int(stats['has_fk'])}  "
                f"size={stats['size_bytes'] / 1e6:.1f}MB  "
                f"growth_rate={growth:.4f}"
            )

    return result


def build_state_tensor(tables):
    """
    Public API called by queryOptimizer.py (and by the feedback worker's
    REINFORCE path).  Returns a float32 tensor of shape [len(tables), 11].
    """
    names   = list(tables)
    stats   = _get_stats_batched(names)
    vectors = [_build_feature_vector(stats[t]) for t in names]
    tensor  = torch.tensor(vectors, dtype=torch.float32)
    print(f"[Vectorizer] Built state tensor: shape={list(tensor.shape)}")
    return tensor
