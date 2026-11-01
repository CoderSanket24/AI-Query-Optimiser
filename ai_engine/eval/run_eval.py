from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
import psycopg2.extras

EVAL_DIR = Path(__file__).resolve().parent
ENGINE_DIR = EVAL_DIR.parent
sys.path.insert(0, str(ENGINE_DIR))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from agent.ppo_agent import ExplainableJoinOptimizer, PPOTrainer, _parse_join_graph, _connected_candidates
try:
    from .workload import ALIAS_TO_TABLE, build_queries, get_templates
except ImportError:
    from workload import ALIAS_TO_TABLE, build_queries, get_templates

SEED = 24681357
TIMEOUT_MS = 50000
ALIASES = {"title": "t", "cast_info": "ci", "name": "n", "movie_keyword": "mk", "keyword": "k", "movie_companies": "mc", "company_name": "cn"}
EDGES = {frozenset(("t", "ci")), frozenset(("ci", "n")), frozenset(("t", "mk")), frozenset(("mk", "k")), frozenset(("t", "mc")), frozenset(("mc", "cn"))}
SETTINGS = {"jit": "off", "synchronize_seqscans": "off", "max_parallel_workers_per_gather": "0", "statement_timeout": str(TIMEOUT_MS), "pg_hint_plan.debug_print": "on", "pg_hint_plan.message_level": "notice"}
RESULTS = EVAL_DIR / "results"
RAW = RESULTS / "raw.csv"
CALIBRATION = RESULTS / "calibration.csv"
BUFFER_COUNTER_VERSION = 2


def parse_top_node_buffers(output):
    """Parse the first plan-node Buffers line, keeping shared and temp counters separate."""
    counts = {"shared_hit_blocks": 0, "shared_read_blocks": 0, "shared_dirtied_blocks": 0,
              "shared_written_blocks": 0, "temp_read_blocks": 0, "temp_written_blocks": 0}
    for line in output.splitlines():
        if "Buffers:" not in line or line.lstrip().startswith(("Planning:", "JIT:", "Execution Time:")):
            continue
        payload = line.split("Buffers:", 1)[1]
        sections = list(re.finditer(r"\b(shared|temp)\s+", payload))
        for index, section in enumerate(sections):
            scope = section.group(1)
            end = sections[index + 1].start() if index + 1 < len(sections) else len(payload)
            counters = payload[section.end():end]
            for counter, value in re.findall(r"\b(hit|read|dirtied|written)=(\d+)", counters):
                if scope == "shared":
                    key = f"shared_{counter}_blocks"
                elif counter in {"read", "written"}:
                    key = f"temp_{counter}_blocks"
                else:
                    continue
                counts[key] = int(value)
        return counts
    return counts


def dsn():
    return {"host": os.getenv("DB_HOST", "localhost"), "port": int(os.getenv("DB_PORT", "5432")), "dbname": os.getenv("DB_NAME", "ai_optimizer"), "user": os.getenv("DB_USERNAME", "postgres"), "password": os.getenv("DB_PASSWORD", "password")}


def set_session(conn, max_parallel):
    with conn.cursor() as cur:
        for key, value in {**SETTINGS, "max_parallel_workers_per_gather": str(max_parallel)}.items():
            cur.execute("SELECT set_config(%s, %s, false)", (key, value))
    conn.commit()


def sql_for(conn, query, order=None):
    sql = query["sql"]
    if order:
        inner = f"({order[0]} {order[1]})"
        for alias in order[2:]:
            inner = f"({inner} {alias})"
        sql = f"/*+ Leading({inner}) */ " + sql
    return conn.cursor().mogrify(sql, query["params"]).decode()


def valid_orders(aliases, edges):
    conditions = [f"{a},{b}" for a, b in edges]
    adj = _parse_join_graph(aliases, conditions)
    valid = []
    for order in itertools.permutations(aliases):
        remaining, placed = list(aliases), set()
        ok = True
        for alias in order:
            candidates = _connected_candidates(remaining, placed, adj)
            if alias not in candidates:
                ok = False
                break
            remaining.remove(alias)
            placed.add(alias)
        if ok:
            valid.append(order)
    return valid


def _state(query):
    from vectorizer.schema_vectorizer import build_state_tensor
    tables = [next(real for real, alias in ALIASES.items() if alias == a) for a in query["tables"]]
    return build_state_tensor(tables)


def make_agents(paths):
    agents = {}
    checkpoint_meta = {}
    for label, path in paths.items():
        payload = __import__("torch").load(path, map_location="cpu", weights_only=True)
        torch = __import__("torch")
        model = ExplainableJoinOptimizer(11, 32)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        agents[label] = (model, PPOTrainer(model))
        checkpoint_meta[label] = {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(), "train_count": payload.get("train_count", 0)}
    return agents, checkpoint_meta


def order_for_agent(trainer, state, aliases, edges, greedy, rng=None):
    torch = __import__("torch")
    adj = _parse_join_graph(aliases, [f"{a},{b}" for a, b in edges])
    with torch.no_grad():
        _, weights = trainer.model(state)
    remaining, placed, order = list(aliases), set(), []
    for _ in aliases:
        choices = _connected_candidates(remaining, placed, adj)
        if greedy:
            index = max(choices, key=lambda a: (float(weights[aliases.index(a)]), -aliases.index(a)))
        else:
            probs = [float(weights[aliases.index(a)]) for a in choices]
            index = (rng or random).choices(choices, weights=probs, k=1)[0]
        order.append(index)
        remaining.remove(index)
        placed.add(index)
    return tuple(order)


def plans_for(query, agents, rng, include_oracle, include_filtered):
    aliases = list(query["tables"])
    edges = [tuple(edge) for edge in EDGES if set(edge).issubset(aliases)]
    plans = [("default", None)]
    if include_oracle and not query.get("quick") and len(aliases) <= 4:
        plans += [("oracle", order) for order in valid_orders(aliases, edges)]
    if not query.get("quick"):
        orders = valid_orders(aliases, edges)
        plans += [("random_k", rng.choice(orders)) for _ in range(3)]
        plans += [("heur_small", order_for_size(query, edges, smallest=True)), ("heur_large", order_for_size(query, edges, smallest=False))]
        if include_filtered:
            plans += [("heur_filtered_small", order_for_size(query, edges, smallest=True, filtered=True))]
        for label, (_, trainer) in agents.items():
            if label == "untrained":
                continue
            suffix = "" if label == "agent" else ":" + label
            state = _state(query)
            plans += [("agent_greedy" + suffix, order_for_agent(trainer, state, aliases, edges, True))]
            plans += [("agent_sampled" + suffix, order_for_agent(trainer, state, aliases, edges, False, rng)) for _ in range(3)]
        if "untrained" in agents:
            plans += [("agent_untrained_greedy", order_for_agent(agents["untrained"][1], _state(query), aliases, edges, True))]
    else:
        plans += [("heur_small", order_for_size(query, edges, smallest=True))]
        plans += [("heur_large", order_for_size(query, edges, smallest=False))]
        for label, (_, trainer) in agents.items():
            if label != "untrained":
                suffix = "" if label == "agent" else ":" + label
                plans += [("agent_greedy" + suffix, order_for_agent(trainer, _state(query), aliases, edges, True))]
    return plans


def order_for_size(query, edges, smallest=True, filtered=False):
    # The required filtered estimate is gathered by single-table EXPLAIN.
    conn = _ACTIVE_CONN[0]
    sizes = {}
    for alias in query["tables"]:
        real = next(name for name, a in ALIASES.items() if a == alias)
        try:
            with conn.cursor() as cur:
                where = query["filter_sql"].get(alias, "") if filtered else ""
                params = query["filter_params"].get(alias, ()) if filtered else ()
                cur.execute("EXPLAIN (FORMAT JSON) SELECT * FROM " + real + where, params)
                sizes[alias] = float(cur.fetchone()[0][0]["Plan"].get("Plan Rows", 1e30))
        except Exception:
            sizes[alias] = 1e30
    aliases = list(query["tables"])
    adj = _parse_join_graph(aliases, [f"{a},{b}" for a, b in edges])
    remaining, placed, order = list(aliases), set(), []
    for _ in aliases:
        choices = _connected_candidates(remaining, placed, adj)
        chosen = (min if smallest else max)(choices, key=lambda a: sizes[a])
        order.append(chosen)
        remaining.remove(chosen)
        placed.add(chosen)
    return tuple(order)


_ACTIVE_CONN = [None]


def notices_used(notices, hint):
    if hint is None:
        return True
    text = "\n".join(notices).lower()
    if "used hint" not in text or "leading" not in text:
        return False
    aliases = re.findall(r"[a-z]+", hint.split("*/", 1)[0].lower())
    return all(alias in text for alias in aliases if alias not in {"leading"})


def explain_fingerprint(conn, statement):
    """Return a stable fingerprint of the planner's physical plan and hint status."""
    conn.notices.clear()
    try:
        with conn.cursor() as cur:
            cur.execute("EXPLAIN (COSTS OFF) " + statement)
            plan_text = "\n".join(str(row[0]) for row in cur.fetchall())
        used = notices_used(conn.notices, statement.split("*/", 1)[0] + "*/" if "/*+ Leading" in statement else None)
        normalized = " ".join(plan_text.split())
        fingerprint = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return fingerprint, used, normalized, ""
    except psycopg2.Error as exc:
        conn.rollback()
        return "", False, "", str(exc)


def explain_cost(conn, statement):
    conn.notices.clear()
    with conn.cursor() as cur:
        cur.execute("EXPLAIN (FORMAT JSON) " + statement)
        plan = cur.fetchone()[0][0]["Plan"]
    used = notices_used(conn.notices, statement.split("*/", 1)[0] + "*/" if "/*+ Leading" in statement else None)
    return float(plan.get("Total Cost", 0.0)), used


def execution(conn, statement):
    conn.notices.clear()
    start = time.perf_counter()
    try:
        with conn.cursor() as cur:
            cur.execute("EXPLAIN (ANALYZE, BUFFERS, TIMING OFF) " + statement)
            output = "\n".join(row[0] for row in cur.fetchall())
        elapsed = (time.perf_counter() - start) * 1000
        match = re.search(r"Execution Time:\s*([0-9.]+)\s*ms", output)
        buffer_counts = parse_top_node_buffers(output)
        return (float(match.group(1)) if match else elapsed, elapsed, False,
                notices_used(conn.notices, "/*+ Leading" in statement and statement.split("*/", 1)[0] + "*/" if "/*+ Leading" in statement else None), "",
                *(buffer_counts[key] for key in ("shared_hit_blocks", "shared_read_blocks", "shared_dirtied_blocks",
                                                   "shared_written_blocks", "temp_read_blocks", "temp_written_blocks")))
    except psycopg2.Error as exc:
        elapsed = (time.perf_counter() - start) * 1000
        timeout = getattr(exc, "pgcode", None) == "57014"
        conn.rollback()
        return (TIMEOUT_MS if timeout else elapsed, elapsed, timeout, False, str(exc), "", "", "", "", "", "")


def filter_profile(conn, query, table_totals):
    """Count each filtered relation, then identify its most selective driving filter."""
    profiles = []
    for alias in query["filter_aliases"]:
        table = ALIAS_TO_TABLE[alias]
        where = query["filter_sql"][alias]
        params = query["filter_params"][alias]
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM " + table + where, params)
                filtered_count = int(cur.fetchone()[0])
            if table not in table_totals:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM " + table)
                    table_totals[table] = int(cur.fetchone()[0])
            total_count = table_totals[table]
            fraction = filtered_count / total_count if total_count else 0.0
            profiles.append({"alias": alias, "table": table, "filter": where.strip(), "filtered_row_count": filtered_count,
                             "table_row_count": total_count, "filtered_fraction": fraction})
        except psycopg2.Error as exc:
            conn.rollback()
            profiles.append({"alias": alias, "table": table, "filter": where.strip(), "filtered_row_count": "",
                             "table_row_count": table_totals.get(table, ""), "filtered_fraction": "", "error": str(exc)})
    numeric = [item for item in profiles if isinstance(item["filtered_fraction"], (int, float))]
    driver = min(numeric, key=lambda item: (item["filtered_fraction"], item["alias"])) if numeric else None
    empty = any(item.get("filtered_row_count") == 0 for item in profiles)
    over_half = any(item["filtered_fraction"] > 0.5 for item in numeric)
    return {
        "driving_table": driver["table"] if driver else "",
        "driving_filter": driver["filter"] if driver else "",
        "filtered_row_count": driver["filtered_row_count"] if driver else "",
        "table_row_count": driver["table_row_count"] if driver else "",
        "filtered_fraction": driver["filtered_fraction"] if driver else "",
        "empty_filter": empty,
        "selects_over_50_percent": over_half,
        "filter_counts_json": json.dumps(profiles, sort_keys=True),
        "filter_count_error": "; ".join(item.get("error", "") for item in profiles if item.get("error")),
    }


def select_quick_queries(queries, requested):
    """Use a deterministic, representative six-template minimum in quick mode."""
    priority = ["T01", "T04", "T06", "T07", "T10", "T12"]
    by_template = {}
    for query in queries:
        by_template.setdefault(query["template_id"], []).append(query)
    chosen = []
    for template_id in priority:
        if template_id in by_template:
            chosen.append(by_template[template_id][0])
    target = max(6, requested)
    for query in queries:
        if len(chosen) >= target:
            break
        if query["template_id"] not in {item["template_id"] for item in chosen}:
            chosen.append(query)
    chosen_ids = {item["query_id"] for item in chosen}
    return [query for query in queries if query["query_id"] in chosen_ids]


RAW_FIELDS = ["query_id", "template_id", "split", "n_tables", "table_set", "arm", "plan_id", "plan_fingerprint",
              "order", "repeat", "phase", "exec_ms", "wall_ms", "shared_hit_blocks", "shared_read_blocks",
              "shared_dirtied_blocks", "shared_written_blocks", "temp_read_blocks", "temp_written_blocks",
              "buffer_parser_version", "timeout", "hint_used", "hint_reused", "reuse_source", "error"]


def ensure_raw_schema():
    """Expand an older raw.csv header without discarding its historical rows."""
    if not RAW.exists():
        return
    with RAW.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames == RAW_FIELDS:
            return
        old_rows = list(reader)
    temp = RAW.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=RAW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for old in old_rows:
            row = {field: old.get(field, "") for field in RAW_FIELDS}
            if not row["table_set"]:
                row["table_set"] = ""
            writer.writerow(row)
    os.replace(temp, RAW)


def plan_variants(conn, queries, plan_map, with_cost):
    """Explain every candidate hint and map it to a physical plan fingerprint."""
    dry_rows, grouped = [], {}
    for query in queries:
        plain = sql_for(conn, query)
        plain_fp, plain_used, _, plain_error = explain_fingerprint(conn, plain)
        if with_cost and not plain_error:
            try:
                plain_cost, _ = explain_cost(conn, plain)
            except psycopg2.Error as exc:
                conn.rollback(); plain_cost, plain_error = "", str(exc)
        else:
            plain_cost = ""
        if with_cost:
            dry_rows.append({"query_id": query["query_id"], "template_id": query["template_id"], "arm": "query_validation",
                             "order": "", "hint_used": plain_used, "estimated_total_cost": plain_cost,
                             "plan_fingerprint": plain_fp, "error": plain_error})
        for arm, order in plan_map[query["query_id"]]:
            statement = sql_for(conn, query, order)
            fingerprint, used, _, error = explain_fingerprint(conn, statement)
            cost = ""
            if with_cost and not error:
                try:
                    cost, cost_hint_used = explain_cost(conn, statement)
                    used = used and cost_hint_used
                except psycopg2.Error as exc:
                    conn.rollback(); error = str(exc)
            variant = {"query": query, "arm": arm, "order": tuple(order) if order else (), "statement": statement,
                       "hint_used": bool(used), "plan_fingerprint": fingerprint, "error": error, "cost": cost}
            if fingerprint:
                key = (query["query_id"], fingerprint)
                group = grouped.setdefault(key, {"query": query, "fingerprint": fingerprint, "variants": {}})
                group["variants"].setdefault(arm, variant)
            if with_cost:
                dry_rows.append({"query_id": query["query_id"], "template_id": query["template_id"], "arm": arm,
                                 "order": " ".join(order or ()), "hint_used": used, "estimated_total_cost": cost,
                                 "plan_fingerprint": fingerprint, "error": error})
    return dry_rows, grouped


def write_raw_row(writer, query, fingerprint, arm, variant, repeat, phase, value, hint_reused, reuse_source):
    writer.writerow({"query_id": query["query_id"], "template_id": query["template_id"], "split": query["split"],
                     "n_tables": query["n_tables"], "table_set": query["table_set"], "arm": arm,
                     "plan_id": fingerprint, "plan_fingerprint": fingerprint, "order": " ".join(variant["order"]),
                     "repeat": repeat, "phase": phase, "exec_ms": value[0], "wall_ms": value[1],
                     "shared_hit_blocks": value[5], "shared_read_blocks": value[6],
                     "shared_dirtied_blocks": value[7], "shared_written_blocks": value[8],
                     "temp_read_blocks": value[9], "temp_written_blocks": value[10],
                     "buffer_parser_version": BUFFER_COUNTER_VERSION,
                     "timeout": value[2], "hint_used": variant["hint_used"], "hint_reused": hint_reused,
                     "reuse_source": reuse_source, "error": value[4]})


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ENGINE_DIR.parent, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def ensure_results():
    RESULTS.mkdir(parents=True, exist_ok=True)


def write_metadata(conn, checkpoint_meta, max_parallel, note="", extra_unseen=False):
    import numpy, scipy, torch
    ensure_results()
    with conn.cursor() as cur:
        cur.execute("SELECT version()")
        pg_version = cur.fetchone()[0]
    old = {}
    path = RESULTS / "metadata.json"
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            old = {}
    started = old.get("started_at_utc", datetime.now(timezone.utc).isoformat()) if old.get("workload_version") == 2 else datetime.now(timezone.utc).isoformat()
    effective_note = note or (old.get("note", "") if old.get("workload_version") == 2 else "")
    metadata = {"workload_version": 2, "git_commit": git_hash(), "postgres_version": pg_version,
                "settings": {**SETTINGS, "max_parallel_workers_per_gather": str(max_parallel)}, "seed": SEED,
                "checkpoint": checkpoint_meta, "note": effective_note, "extra_unseen": bool(extra_unseen),
                "library_versions": {"numpy": numpy.__version__, "scipy": scipy.__version__, "torch": torch.__version__, "psycopg2": psycopg2.__version__},
                "started_at_utc": started, "timestamps_utc": old.get("timestamps_utc", {}) if old.get("workload_version") == 2 else {}}
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def estimate_only():
    """Estimate the full run from saved EXPLAIN plans and timings; never connects to PostgreSQL."""
    dry_path = RESULTS / "dry_run.csv"
    if not dry_path.exists():
        raise SystemExit(f"--estimate-only needs the full dry-run cache at {dry_path}; run --dry-run first.")
    with dry_path.open(newline="", encoding="utf-8") as f:
        dry_rows = list(csv.DictReader(f))
    present_templates = {row.get("template_id") for row in dry_rows if row.get("arm") == "query_validation"}
    missing_base = {f"T{i:02d}" for i in range(1, 13)} - present_templates
    missing_extra = {"T13", "T14"} - present_templates
    has_oracle = any(row.get("arm") == "oracle" for row in dry_rows)
    if missing_base or missing_extra or not has_oracle:
        missing = []
        if missing_base:
            missing.append("base templates: " + ", ".join(sorted(missing_base)))
        if missing_extra:
            missing.append("extra-unseen templates: " + ", ".join(sorted(missing_extra)))
        if not has_oracle:
            missing.append("oracle variants")
        raise SystemExit("--estimate-only requires a superset dry-run cache containing " + "; ".join(missing) + ". Run --dry-run --extra-unseen --oracle first.")

    raw_path = RESULTS / "raw.csv"
    quick_times = {}
    if raw_path.exists():
        with raw_path.open(newline="", encoding="utf-8") as f:
            raw_rows = list(csv.DictReader(f))
        seen = set()
        for row in raw_rows:
            if row.get("phase") != "measured" or not row.get("plan_fingerprint"):
                continue
            key = (row.get("query_id"), row.get("plan_fingerprint"), row.get("repeat"))
            if key in seen:
                continue
            seen.add(key)
            quick_times.setdefault(key[:2], []).append((float(row.get("exec_ms") or 0), row.get("timeout", "").lower() == "true"))
    quick_medians = {key: (float(__import__("statistics").median(value for value, _ in values)), all(timeout for _, timeout in values))
                     for key, values in quick_times.items()}

    calibration = {}
    cal_path = RESULTS / "calibration.csv"
    if cal_path.exists():
        with cal_path.open(newline="", encoding="utf-8") as f:
            calibration = {row["query_id"]: row for row in csv.DictReader(f)}
    table_counts = {template["id"]: len(template["tables"]) for template in get_templates(True)}
    calibration_by_size = defaultdict(list)
    for row in calibration.values():
        if row.get("default_exec_ms"):
            calibration_by_size[int(row.get("n_tables") or 0)].append(
                (float(row["default_exec_ms"]), row.get("timeout", "").lower() == "true"))
    size_fallback = {}
    for size, values in calibration_by_size.items():
        size_fallback[size] = (float(__import__("statistics").median(v for v, _ in values)),
                               all(timeout for _, timeout in values))

    def configuration(extra, oracle):
        pairs = {}
        for row in dry_rows:
            arm, query_id = row.get("arm", ""), row.get("query_id", "")
            if arm == "query_validation" or (arm == "oracle" and not oracle):
                continue
            if query_id.startswith(("T13-", "T14-")) and not extra:
                continue
            fp = row.get("plan_fingerprint", "")
            if fp and not row.get("error"):
                pairs[(query_id, fp)] = row
        return pairs

    scenarios = [
        ("full baseline", False, False),
        ("full + --extra-unseen", True, False),
        ("full + --oracle", False, True),
        ("full + --extra-unseen --oracle", True, True),
    ]
    print("Estimate-only: offline calculation from dry_run.csv, raw.csv, and calibration.csv; no PostgreSQL connection or SQL execution.")
    for label, extra, oracle in scenarios:
        pairs = configuration(extra, oracle)
        if not pairs:
            print(f"{label}: unavailable (no matching plan fingerprints)")
            continue
        expected_ms, from_quick, from_cal, fallback = 0.0, 0, 0, 0
        for query_id, fingerprint in pairs:
            if (query_id, fingerprint) in quick_medians:
                per_call, was_timeout = quick_medians[(query_id, fingerprint)]
                from_quick += 1
            else:
                row = calibration.get(query_id)
                if row and row.get("default_exec_ms"):
                    per_call = min(TIMEOUT_MS, float(row["default_exec_ms"]))
                    was_timeout = row.get("timeout", "").lower() == "true"
                    from_cal += 1
                else:
                    template_id = query_id.split("-", 1)[0]
                    per_call, was_timeout = size_fallback.get(table_counts.get(template_id, 0), (TIMEOUT_MS, True))
                    per_call = min(TIMEOUT_MS, per_call)
                    fallback += 1
                if was_timeout:
                    per_call = TIMEOUT_MS
            repeat_count = 0 if was_timeout else (3 if per_call > 10000 else 5)
            expected_ms += per_call * (1 + repeat_count)
        count = len(pairs)
        worst_hours = count * 6 * TIMEOUT_MS / 3_600_000
        print(f"{label}: unique query/physical-plan pairs={count}; worst-case={worst_hours:.2f} h ({count} x 6 x {TIMEOUT_MS} ms); expected={expected_ms / 60000:.2f} min; estimates from quick per-plan medians={from_quick}, calibration default-plan fallback={from_cal}, table-size fallback={fallback}.")
    print("Assumptions: one warm-up per unique plan; full-run R=5, reduced to R=3 when the estimated warm-up exceeds 10 s; timeout warm-ups skip repeats. Unmeasured plans use that query's calibrated default time, or the median calibration for the same table count when no query calibration exists. EXPLAIN overhead and data/cache drift are excluded; counts are physical-plan fingerprints from the cached EXPLAIN dry-run.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", action="append", default=[], help="label=checkpoint.pt")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--heur-filtered", action="store_true")
    parser.add_argument("--extra-unseen", action="store_true", help="include T13/T14 test-only table sets")
    parser.add_argument("--note", default="", help="text copied to metadata.json and summary.md")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-dry-run", action="store_true")
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--estimate-only", action="store_true", help="print cached full-run estimates offline; no database access")
    parser.add_argument("--max-parallel-workers", type=int, default=0)
    parser.add_argument("--quick-queries", type=int, default=6)
    args = parser.parse_args()
    if args.estimate_only:
        estimate_only()
        return
    import torch
    random.seed(SEED); torch.manual_seed(SEED)
    paths = dict(item.split("=", 1) for item in args.agent)
    if not paths:
        paths = {"agent": str(ENGINE_DIR / "checkpoints" / "ppo_checkpoint.pt")}
    agents, checkpoint_meta = make_agents(paths)
    torch.manual_seed(SEED + 1)
    untrained_model = ExplainableJoinOptimizer(11, 32); untrained_model.eval()
    agents["untrained"] = (untrained_model, PPOTrainer(untrained_model))
    conn = psycopg2.connect(**dsn()); _ACTIVE_CONN[0] = conn
    try:
        set_session(conn, args.max_parallel_workers)
        all_queries = build_queries(conn, extra_unseen=args.extra_unseen)
        queries = select_quick_queries(all_queries, args.quick_queries) if args.quick else all_queries
        for query in queries:
            query["quick"] = args.quick
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM pg_stat_activity WHERE datname=current_database() AND state='active' AND pid<>pg_backend_pid()")
            other_active = cur.fetchone()[0]
        print(f"Active other sessions: {other_active}" + (" WARNING: other queries may affect timings." if other_active else ""))
        settings = {**SETTINGS, "max_parallel_workers_per_gather": str(args.max_parallel_workers)}
        print("Session settings:", json.dumps(settings, sort_keys=True))
        ensure_results()
        metadata = write_metadata(conn, checkpoint_meta, args.max_parallel_workers, args.note, args.extra_unseen)
        print("Checkpoint metadata:", json.dumps(checkpoint_meta, sort_keys=True))
        rng = random.Random(SEED)
        plan_map = {query["query_id"]: plans_for(query, agents, rng, args.oracle, args.heur_filtered) for query in queries}

        if not args.skip_dry_run:
            dry_rows, grouped = plan_variants(conn, queries, plan_map, with_cost=True)
            dry_path = RESULTS / ("dry_run_quick.csv" if args.quick else "dry_run.csv")
            fields = ["query_id", "template_id", "arm", "order", "hint_used", "estimated_total_cost", "plan_fingerprint", "error"]
            with dry_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(dry_rows)
            metadata["timestamps_utc"]["dry_run"] = datetime.now(timezone.utc).isoformat()
            (RESULTS / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(f"Dry-run: {len(dry_rows)} query-validation and (query,hint) EXPLAIN records; {sum(len(g['variants']) for g in grouped.values())} variants map to {len(grouped)} distinct physical plans; output={dry_path}")
            for query in queries:
                print(f"DRY {query['query_id']} params={query['params']}")
        else:
            dry_rows, grouped = [], {}

        if args.dry_run:
            conn.close(); return

        if not args.skip_calibration:
            set_session(conn, args.max_parallel_workers)
            fields = ["query_id", "template_id", "split", "table_set", "n_tables", "params", "driving_table", "driving_filter",
                      "filtered_row_count", "table_row_count", "filtered_fraction", "empty_filter", "selects_over_50_percent",
                      "filter_counts_json", "filter_count_error", "default_exec_ms", "wall_ms", "timeout", "in_20ms_20s_band", "error"]
            table_totals = {}
            with CALIBRATION.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
                for query in queries:
                    profile = filter_profile(conn, query, table_totals)
                    value = execution(conn, sql_for(conn, query))
                    in_band = not value[2] and 20 <= value[0] <= 20000
                    row = {"query_id": query["query_id"], "template_id": query["template_id"], "split": query["split"],
                           "table_set": query["table_set"], "n_tables": query["n_tables"], "params": json.dumps(query["params"]),
                           **profile, "default_exec_ms": value[0], "wall_ms": value[1], "timeout": value[2],
                           "in_20ms_20s_band": in_band, "error": value[4]}
                    writer.writerow(row); f.flush()
                    print(f"CAL {query['query_id']} params={query['params']} driver={profile['driving_table']} rows={profile['filtered_row_count']}/{profile['table_row_count']} fraction={profile['filtered_fraction']} empty={profile['empty_filter']} over50={profile['selects_over_50_percent']} default_ms={value[0]:.3f} timeout={value[2]} in_band={in_band}")
                    if profile["empty_filter"]:
                        print(f"CALIBRATION EMPTY FILTER: {query['query_id']} params={query['params']}")
                    if profile["selects_over_50_percent"]:
                        print(f"CALIBRATION FILTER SELECTS OVER 50%: {query['query_id']} params={query['params']}")
                    if not in_band:
                        print(f"CALIBRATION OUTSIDE 20ms-20s BAND: {query['query_id']} params={query['params']} default_ms={value[0]} timeout={value[2]}")
            metadata["timestamps_utc"]["calibration"] = datetime.now(timezone.utc).isoformat()
            (RESULTS / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(f"Calibration rows: {len(queries)}; output={CALIBRATION}")

        if args.calibrate_only:
            conn.close(); return

        if not grouped:
            # Even with --skip-dry-run, fingerprints are required for safe plan reuse.
            _, grouped = plan_variants(conn, queries, plan_map, with_cost=False)
        if args.skip_dry_run:
            print(f"Plan preflight: {sum(len(g['variants']) for g in grouped.values())} hint variants map to {len(grouped)} distinct physical plans.")

        r = 2 if args.quick else 5
        group_list = list(grouped.values())
        max_repeat = max(r, 3)
        worst_execs = len(group_list) * (1 + max_repeat)
        worst_hours = worst_execs * TIMEOUT_MS / 3_600_000
        calibration = {}
        if CALIBRATION.exists():
            with CALIBRATION.open(newline="", encoding="utf-8") as f:
                calibration = {row["query_id"]: row for row in csv.DictReader(f)}
        realistic_ms, missing_cal = 0.0, 0
        for group in group_list:
            query_id = group["query"]["query_id"]
            row = calibration.get(query_id)
            if row and row.get("default_exec_ms"):
                per_call = min(TIMEOUT_MS, float(row["default_exec_ms"]))
                if row.get("timeout", "").lower() == "true":
                    expected_calls = 1  # timeout warm-up causes measured repeats to be skipped
                else:
                    expected_calls = 1 + (3 if per_call > 10000 else r)
            else:
                per_call = TIMEOUT_MS
                missing_cal += 1
                expected_calls = 1 + r
            realistic_ms += per_call * expected_calls
        print(f"WORST-CASE runtime estimate: {len(group_list)} unique (query, physical-plan) pairs x {1 + max_repeat} calls x {TIMEOUT_MS} ms timeout = {worst_hours:.2f} hours; assumes every call reaches the timeout and allows 3 repeats after a >10s warm-up.")
        print(f"EXPECTED runtime estimate: {realistic_ms/1000/60:.2f} minutes from per-query calibration times; repeat counts use calibration >10s for R=3 and otherwise the requested R={r}; timeout warm-ups count once and skip repeats. {missing_cal} plan-query estimates use the 50s timeout fallback; assumes hint plans take default-plan time.")

        ensure_raw_schema()
        existing_rows = []
        if RAW.exists():
            with RAW.open(newline="", encoding="utf-8") as f:
                existing_rows = list(csv.DictReader(f))
        run_ids = {query["query_id"] for query in queries}
        stale_queries = {row.get("query_id") for row in existing_rows
                         if row.get("query_id") in run_ids
                         and row.get("buffer_parser_version") != str(BUFFER_COUNTER_VERSION)}
        if stale_queries:
            # Rows written by an older parser cannot safely resume: its read
            # regex missed PostgreSQL's shared hit=... read=... format.
            existing_rows = [row for row in existing_rows if row.get("query_id") not in stale_queries]
            with RAW.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=RAW_FIELDS)
                writer.writeheader(); writer.writerows(existing_rows)
            print(f"Refreshing {len(stale_queries)} query result slices with stale/missing BUFFERS parsing: {', '.join(sorted(stale_queries))}")
        physical_done = {}
        written_keys = set()
        for row in existing_rows:
            q, fp, repeat = row.get("query_id", ""), row.get("plan_fingerprint", ""), row.get("repeat", "")
            if not fp:
                continue
            written_keys.add((q, fp, repeat, row.get("arm", "")))
            key = (q, fp, repeat)
            if key not in physical_done or row.get("hint_reused", "").lower() != "true":
                physical_done[key] = row

        mode = "a" if RAW.exists() else "w"
        if CALIBRATION.exists():
            with CALIBRATION.open(newline="", encoding="utf-8") as f:
                calibration = {row["query_id"]: row for row in csv.DictReader(f)}

        def stored_value(row):
            return (float(row.get("exec_ms") or 0), float(row.get("wall_ms") or 0),
                    row.get("timeout", "").lower() == "true", row.get("hint_used", "").lower() == "true", row.get("error", ""),
                    *(row.get(key, "") for key in ("shared_hit_blocks", "shared_read_blocks", "shared_dirtied_blocks",
                                                     "shared_written_blocks", "temp_read_blocks", "temp_written_blocks")))

        def append_group(writer, group, repeat, phase, value, owner_arm):
            query, fp = group["query"], group["fingerprint"]
            for arm, variant in group["variants"].items():
                key = (query["query_id"], fp, str(repeat), arm)
                if key in written_keys:
                    continue
                write_raw_row(writer, query, fp, arm, variant, repeat, phase, value, arm != owner_arm, owner_arm if arm != owner_arm else "")
                written_keys.add(key)

        ensure_raw_schema()
        mode = "a" if RAW.exists() else "w"
        with RAW.open(mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RAW_FIELDS)
            if mode == "w": writer.writeheader()
            rng.shuffle(group_list)
            for group in group_list:
                variants = group["variants"]
                owner = next((arm for arm, variant in variants.items() if variant["hint_used"]), next(iter(variants)))
                group["owner"] = owner
                warm_key = (group["query"]["query_id"], group["fingerprint"], "warmup")
                if warm_key in physical_done:
                    warm = stored_value(physical_done[warm_key])
                    print(f"RESUME {group['query']['query_id']} {group['fingerprint'][:12]} warmup already recorded")
                else:
                    warm = execution(conn, variants[owner]["statement"])
                    append_group(writer, group, "warmup", "warmup", warm, owner); f.flush()
                    physical_done[warm_key] = {"exec_ms": warm[0], "wall_ms": warm[1], "timeout": str(warm[2]), "hint_used": str(warm[3]), "error": warm[4],
                                               **dict(zip(("shared_hit_blocks", "shared_read_blocks", "shared_dirtied_blocks", "shared_written_blocks", "temp_read_blocks", "temp_written_blocks"), warm[5:11])),
                                               "buffer_parser_version": BUFFER_COUNTER_VERSION, "hint_reused": "false"}
                group["warm"] = warm
                group["repeat_count"] = 3 if (not warm[2] and warm[0] > 10000) else r
                if warm[2]:
                    for repeat in range(1, group["repeat_count"] + 1):
                        key = (group["query"]["query_id"], group["fingerprint"], str(repeat))
                        value = stored_value(physical_done[key]) if key in physical_done else (TIMEOUT_MS, 0, True, warm[3], "warm-up timed out; repeat skipped", "", "", "", "", "", "")
                        append_group(writer, group, repeat, "skipped_after_warmup_timeout", value, owner); f.flush()

            for repeat in range(1, max((group["repeat_count"] for group in group_list), default=r) + 1):
                block = [group for group in group_list if group["repeat_count"] >= repeat and not group["warm"][2]]
                rng.shuffle(block)
                for group in block:
                    query, fp, owner = group["query"], group["fingerprint"], group["owner"]
                    key = (query["query_id"], fp, str(repeat))
                    if key in physical_done:
                        value = stored_value(physical_done[key])
                    else:
                        value = execution(conn, group["variants"][owner]["statement"])
                        physical_done[key] = {"exec_ms": value[0], "wall_ms": value[1], "timeout": str(value[2]), "hint_used": str(value[3]), "error": value[4],
                                              **dict(zip(("shared_hit_blocks", "shared_read_blocks", "shared_dirtied_blocks", "shared_written_blocks", "temp_read_blocks", "temp_written_blocks"), value[5:11])),
                                              "buffer_parser_version": BUFFER_COUNTER_VERSION, "hint_reused": "false"}
                    append_group(writer, group, repeat, "measured", value, owner); f.flush()

        metadata["timestamps_utc"]["evaluation_finished"] = datetime.now(timezone.utc).isoformat()
        metadata["quick_mode"] = bool(args.quick)
        metadata["queries_in_run"] = [query["query_id"] for query in queries]
        metadata["unique_query_plan_pairs"] = len(grouped)
        (RESULTS / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        conn.close()
        print(f"Evaluation rows appended/resumed in {RAW}; unique physical plans measured once; calibration={CALIBRATION}")
    finally:
        if not conn.closed:
            conn.close()


if __name__ == "__main__":
    main()
