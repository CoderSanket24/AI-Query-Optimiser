"""
bench_latency.py
----------------
Benchmarks the full JoinSense pipeline end-to-end by cycling through the
QUERIES list from run_queries.py.

Usage:
    python bench_latency.py [N]

    N  — number of queries to fire (default 200, cycling through QUERIES).

Warm-up:
    The first 10 queries are sent but excluded from all statistics and the CSV.
    This lets the JVM JIT compile hot paths and lets the PPO model settle.
    The warm-up count is printed at the top of the output.

Failure handling:
    HTTP errors and timeouts are recorded with status != "ok" and kept in the
    CSV.  Only status=="ok" rows are included in the P50/P95 statistics.
    The script never crashes on a single bad query.

Output files:
    latency_breakdown.csv  — one row per non-warm-up query

Statistics printed:
    Overall P50 / P95 per stage, and the P50 share of total_ms.
    Then the same split by n_tables (proxy for query complexity).

Stages collected from latency_breakdown_ms in the response:
    Java-side  : firewall_ms, extract_ms, optimize_http_ms, active_conn_query_ms,
                 pool_wait_ms, blks_before_query_ms, set_timeout_ms, explain_call_ms,
                 exec_time_ms_gap, blks_after_query_ms, reset_ms, feedback_http_ms,
                 total_ms
    /optimize  : opt_vectorize, opt_sample, opt_attention, opt_hint_build,
                 opt_optimize_total
    /feedback  : fb_anomaly_predict, fb_training, fb_checkpoint_save,
                 fb_analytics, fb_forest_refit, fb_feedback_total
    Derived    : unaccounted_ms, optimize_hop_overhead_ms, feedback_hop_overhead_ms
"""

import sys
import json
import csv
import time
import urllib.request
import urllib.error

# Import shared constants from run_queries.py (same directory).
# QUERIES is the benchmark workload; URL, HEADERS, HTTP_TIMEOUT_SEC are reused.
from run_queries import QUERIES, URL, HEADERS, HTTP_TIMEOUT_SEC

WARM_UP = 10

# Ordered list of all stage keys expected in latency_breakdown_ms.
# Missing keys (e.g. server_timings_ms absent on older server) are recorded as None.
STAGES = [
    # Java pipeline stages
    "firewall_ms",
    "extract_ms",
    "optimize_http_ms",
    "active_conn_query_ms",
    "pool_wait_ms",
    "blks_before_query_ms",
    "set_timeout_ms",
    "explain_call_ms",
    "exec_time_ms_gap",
    "blks_after_query_ms",
    "reset_ms",
    "feedback_http_ms",
    "total_ms",
    # /optimize server timings
    "opt_vectorize",
    "opt_sample",
    "opt_attention",
    "opt_hint_build",
    "opt_optimize_total",
    # /feedback server timings
    "fb_anomaly_predict",
    "fb_training",
    "fb_checkpoint_save",
    "fb_analytics",
    "fb_forest_refit",
    "fb_feedback_total",
    # Derived
    "unaccounted_ms",
    "optimize_hop_overhead_ms",
    "feedback_hop_overhead_ms",
]

CSV_FIELDS = ["query_index", "n_tables", "status", "sql_truncated"] + STAGES


def n_tables_from_sql(sql: str) -> int:
    """
    Count the number of tables referenced by counting JOIN keywords + 1.
    This is a fast proxy for query complexity — good enough for grouping.
    """
    return sql.upper().count(" JOIN ") + 1


def _percentile(values: list[float], p: float) -> float:
    """
    Return the p-th percentile (0 ≤ p ≤ 100) of a non-empty list.
    Uses linear interpolation between adjacent ranks.
    """
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = lo + 1
    if hi >= len(s):
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _print_stage_table(rows: list[dict]) -> None:
    """
    Print a formatted P50 / P95 / share% table for the given row set.
    Only rows with status=="ok" are used; stages with no data are skipped.
    """
    if not rows:
        print("  (no successful queries)")
        return

    total_vals = [r["total_ms"] for r in rows
                  if r.get("total_ms") is not None]
    median_total = _percentile(total_vals, 50) if total_vals else 1.0

    print(f"  {'stage':<30}  {'P50 ms':>10}  {'P95 ms':>10}  {'share%':>8}")
    print("  " + "-" * 64)

    for stage in STAGES:
        vals = [r[stage] for r in rows
                if r.get(stage) is not None and isinstance(r[stage], (int, float))]
        if not vals:
            continue
        p50   = _percentile(vals, 50)
        p95   = _percentile(vals, 95)
        share = (p50 / median_total * 100.0) if median_total > 0 else 0.0
        print(f"  {stage:<30}  {p50:>10.2f}  {p95:>10.2f}  {share:>7.1f}%")


def run_benchmark(n: int = 200) -> None:
    print(f"bench_latency.py  N={n}  warm_up={WARM_UP}  url={URL}")
    print(f"Cycling through {len(QUERIES)} queries in QUERIES list.")
    print(f"First {WARM_UP} queries are warm-up and are excluded from statistics.\n")

    all_rows: list[dict] = []   # non-warm-up rows only

    for i in range(n):
        sql     = QUERIES[i % len(QUERIES)]
        n_tabs  = n_tables_from_sql(sql)
        is_wu   = (i < WARM_UP)
        label   = f"[{i + 1:04d}/{n}]{'(warm-up)' if is_wu else '         '}"

        payload = json.dumps({"query": sql}).encode("utf-8")
        req     = urllib.request.Request(URL, data=payload, headers=HEADERS, method="POST")

        status    = "ok"
        breakdown = {}

        try:
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SEC) as resp:
                body      = json.loads(resp.read())
                breakdown = body.get("latency_breakdown_ms") or {}
        except urllib.error.HTTPError as exc:
            status = f"http_{exc.code}"
            print(f"{label} HTTP {exc.code} {exc.reason}")
        except urllib.error.URLError as exc:
            # Covers timeouts (urllib wraps socket.timeout as URLError on some builds)
            reason = str(exc.reason)
            status = "timeout" if "timed out" in reason.lower() else "url_error"
            print(f"{label} {status.upper()}: {reason}")
        except Exception as exc:
            status = "error"
            print(f"{label} ERROR ({type(exc).__name__}): {exc}")

        total_ms = breakdown.get("total_ms")
        print(f"{label} status={status}  n_tables={n_tabs}  total_ms={total_ms}")

        if not is_wu:
            row: dict = {
                "query_index":  i,
                "n_tables":     n_tabs,
                "status":       status,
                "sql_truncated": sql[:60],
            }
            for stage in STAGES:
                row[stage] = breakdown.get(stage)  # None if missing
            all_rows.append(row)

    # ── Write CSV ────────────────────────────────────────────────────
    csv_path = "latency_breakdown.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nCSV written → {csv_path}  ({len(all_rows)} rows, warm-up excluded)\n")

    # ── Statistics (ok rows only) ────────────────────────────────────
    ok_rows = [r for r in all_rows if r["status"] == "ok"]
    n_ok    = len(ok_rows)
    n_fail  = len(all_rows) - n_ok

    print("=" * 70)
    print(f"Results (excl. warm-up):  {n_ok} ok  |  {n_fail} failed/timeout")
    print("=" * 70)

    print("\n── Overall P50 / P95 per stage ──────────────────────────────────────")
    _print_stage_table(ok_rows)

    # Split by n_tables
    all_n_tabs = sorted({r["n_tables"] for r in ok_rows})
    for nt in all_n_tabs:
        subset = [r for r in ok_rows if r["n_tables"] == nt]
        print(f"\n── n_tables={nt}  (n={len(subset)}) ──────────────────────────────────")
        _print_stage_table(subset)

    print("\nDone.")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    run_benchmark(n)
