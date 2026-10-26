"""
check_vectorizer_equivalence.py
-------------------------------
Verifies that the batched schema-stat fetch in vectorizer/schema_vectorizer.py
produces feature vectors that are EXACTLY equal to the legacy per-table fetch,
then benchmarks cold builds (cache cleared) for 2/3/4-table sets, old vs new.

The LEGACY per-table implementation lives here (verbatim from the pre-batching
module) as the reference implementation — it is NOT in the production module.

Run from ai_engine/ inside the project venv (needs torch + psycopg2 + a live
PostgreSQL database reachable at _DB_CONFIG):

    python tests/check_vectorizer_equivalence.py
"""

import os
import re
import sys
import time
import traceback

# Make `ai_engine` importable regardless of how the test is launched.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEPS_OK = True
DEPS_ERR = None
try:
    import psycopg2
    import torch
    from vectorizer import schema_vectorizer as sv
    from run_queries import QUERIES
except Exception as exc:  # pragma: no cover
    DEPS_OK = False
    DEPS_ERR = exc


# ── Legacy per-table reference (verbatim from the pre-batching module) ──
_LEGACY_CACHE = {}


def _fetch_stats_legacy(table_name):
    """Original per-table fetch (6 queries on its own connection)."""
    conn = psycopg2.connect(**sv._DB_CONFIG)
    try:
        cur = conn.cursor()

        cur.execute("""
            SELECT reltuples, relpages
            FROM   pg_class
            WHERE  relname = %s AND relkind = 'r';
        """, (table_name,))
        row = cur.fetchone()
        row_count  = float(row[0]) if row and row[0] > 0 else 1.0
        page_count = float(row[1]) if row and row[1] > 0 else 1.0

        cur.execute("""
            SELECT COUNT(*)
            FROM   information_schema.columns
            WHERE  table_schema = 'public' AND table_name = %s;
        """, (table_name,))
        col_count = float(cur.fetchone()[0] or 1)

        cur.execute("""
            SELECT COUNT(*)
            FROM   pg_indexes
            WHERE  schemaname = 'public' AND tablename = %s;
        """, (table_name,))
        index_count = float(cur.fetchone()[0] or 0)

        cur.execute("""
            SELECT COUNT(*)
            FROM   information_schema.table_constraints
            WHERE  table_schema = 'public'
              AND  table_name = %s
              AND  constraint_type = 'PRIMARY KEY';
        """, (table_name,))
        has_pk = 1.0 if cur.fetchone()[0] > 0 else 0.0

        cur.execute("""
            SELECT COUNT(*)
            FROM   information_schema.table_constraints
            WHERE  table_schema = 'public'
              AND  table_name = %s
              AND  constraint_type = 'FOREIGN KEY';
        """, (table_name,))
        has_fk = 1.0 if cur.fetchone()[0] > 0 else 0.0

        # NOTE: raises when the relation does not exist (text -> regclass cast),
        # which is why a missing table falls back to the neutral dict.
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
        print(f"[Vectorizer][legacy] DB error for table '{table_name}': {e}")
        return dict(sv._NEUTRAL_STATS)
    finally:
        conn.close()


def _get_stats_cached_legacy(table_name):
    now = time.time()

    if table_name in _LEGACY_CACHE:
        cached_time, cached_stats = _LEGACY_CACHE[table_name]
        if now - cached_time < sv._CACHE_TTL_SECONDS:
            return cached_stats
        prev_row_count = cached_stats["row_count"]
        prev_time      = cached_time
    else:
        prev_row_count = None
        prev_time      = None

    stats = _fetch_stats_legacy(table_name)

    if prev_row_count is not None and prev_time is not None:
        elapsed_seconds  = max(now - prev_time, 1.0)
        row_delta        = stats["row_count"] - prev_row_count
        rows_per_second  = max(row_delta, 0.0) / elapsed_seconds
        growth_rate_norm = min(rows_per_second / sv._MAX_GROWTH_RATE, 1.0)
    else:
        growth_rate_norm = 0.0

    stats["growth_rate_norm"] = growth_rate_norm
    _LEGACY_CACHE[table_name] = (now, stats)
    return stats


def _build_feature_vector_legacy(table_name):
    s = _get_stats_cached_legacy(table_name)

    avg_row_width = s["size_bytes"] / max(s["row_count"], 1)
    row_density   = s["row_count"]  / max(s["page_count"], 1)

    return [
        sv._log_norm(s["row_count"],   sv._MAX_ROWS),
        sv._log_norm(s["page_count"],  sv._MAX_PAGES),
        sv._log_norm(s["col_count"],   sv._MAX_COLS),
        sv._log_norm(s["index_count"], sv._MAX_INDEXES),
        s["has_pk"],
        s["has_fk"],
        sv._log_norm(s["size_bytes"],  sv._MAX_SIZE_BYTES),
        sv._log_norm(avg_row_width,    sv._MAX_ROW_WIDTH),
        min(row_density / sv._MAX_ROW_DENS, 1.0),
        1.0,
        s["growth_rate_norm"],
    ]


def build_state_tensor_legacy(tables):
    vectors = [_build_feature_vector_legacy(t) for t in tables]
    return torch.tensor(vectors, dtype=torch.float32)


# ── Helpers ──────────────────────────────────────────────────────────────

def _tables_from_queries():
    tables = []
    for sql in QUERIES:
        for t in re.findall(r"(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", sql):
            if t not in tables:
                tables.append(t)
    return tables


def _clear_caches():
    sv._CACHE.clear()
    _LEGACY_CACHE.clear()


def _percentile(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = lo + 1
    if hi >= len(s):
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _report_duplicate_relnames(tables):
    """Detect any run_queries table that exists in more than one schema."""
    conn = psycopg2.connect(**sv._DB_CONFIG)
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT c.relname,
                   count(DISTINCT n.nspname)                  AS schema_count,
                   string_agg(DISTINCT n.nspname, ',' ORDER BY n.nspname) AS schemas
            FROM   pg_class c
            JOIN   pg_namespace n ON n.oid = c.relnamespace
            WHERE  c.relkind = 'r' AND c.relname = ANY(%s)
            GROUP BY c.relname
            HAVING count(DISTINCT n.nspname) > 1
            ORDER BY c.relname;
        """, (tables,))
        rows = cur.fetchall()
        if not rows:
            print("duplicate-relname check: none of the tables exist in >1 schema")
        else:
            print("WARNING — relnames present in more than one schema:")
            for relname, cnt, schemas in rows:
                print(f"  {relname}: {cnt} schemas -> {schemas}")
    finally:
        conn.close()


def _compare(run, names, legacy, new):
    ok = torch.equal(legacy, new)
    max_diff = float((legacy - new).abs().max())
    print(f"[run {run}] exact_equal={ok}  max_abs_diff={max_diff:.6e}  "
          f"shape={tuple(new.shape)}")
    if not ok:
        for i, name in enumerate(names):
            lvec = legacy[i]
            nvec = new[i]
            if not torch.equal(lvec, nvec):
                diff_idx = [j for j in range(lvec.numel())
                            if lvec[j].item() != nvec[j].item()]
                print(f"  {name}: differing feature indices {diff_idx}")
                for j in diff_idx:
                    print(f"    [{j}] legacy={lvec[j].item():.9f}  "
                          f"new={nvec[j].item():.9f}")
    return ok


def _time_cold_builds(tables):
    for k in (2, 3, 4):
        ts = tables[:k]
        legacy_ms, new_ms = [], []
        for _ in range(20):
            _clear_caches()
            t0 = time.perf_counter()
            build_state_tensor_legacy(ts)
            legacy_ms.append((time.perf_counter() - t0) * 1000.0)
        for _ in range(20):
            _clear_caches()
            t0 = time.perf_counter()
            sv.build_state_tensor(ts)
            new_ms.append((time.perf_counter() - t0) * 1000.0)
        print(f"{k}-table cold build (n=20):")
        print(f"  legacy  P50={_percentile(legacy_ms, 50):8.3f} ms  "
              f"P95={_percentile(legacy_ms, 95):8.3f} ms")
        print(f"  batched P50={_percentile(new_ms, 50):8.3f} ms  "
              f"P95={_percentile(new_ms, 95):8.3f} ms")


def main() -> int:
    tables = _tables_from_queries()
    print(f"tables ({len(tables)}): {tables}")

    _report_duplicate_relnames(tables)

    # Include a nonexistent table: legacy raises on it -> neutral fallback;
    # batched must produce the SAME fallback (NULL from to_regclass).
    names = tables + ["__definitely_missing_table__"]

    all_ok = True
    for run in (1, 2):
        _clear_caches()
        legacy = build_state_tensor_legacy(names)
        _clear_caches()
        new = sv.build_state_tensor(names)
        all_ok &= _compare(run, names, legacy, new)

    if not all_ok:
        print("RESULT: MISMATCH — batched feature vectors differ from legacy")
        return 1

    print("RESULT: equivalence OK (exact equality)")
    _time_cold_builds(tables)
    return 0


if __name__ == "__main__":
    if not DEPS_OK:
        raise SystemExit(f"Missing dependency ({DEPS_ERR}); run inside the project venv.")
    try:
        raise SystemExit(main())
    except Exception as exc:  # e.g. DB unreachable
        traceback.print_exc()
        print(f"ERROR: {type(exc).__name__}: {exc}")
        print("Is PostgreSQL reachable at sv._DB_CONFIG and is it populated?")
        raise SystemExit(2)
