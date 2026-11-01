from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

BASE = Path(__file__).resolve().parent
RAW = BASE / "results" / "raw.csv"
OUT = BASE / "results" / "summary.md"
SEED = 7654321

try:
    from .workload import get_templates
except ImportError:
    from workload import get_templates


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ci_geom(pairs, cluster, draws=10000, seed=SEED):
    if not pairs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    clusters = sorted({cluster[q] for q, _ in pairs})
    by_cluster = {c: [ratio for q, ratio in pairs if cluster[q] == c] for c in clusters}
    estimates = []
    for _ in range(draws):
        sampled = [rng.choice(clusters) for _ in clusters]
        ratios = [x for c in sampled for x in by_cluster[c]]
        estimates.append(math.exp(sum(math.log(max(x, 1e-12)) for x in ratios) / len(ratios)))
    return tuple(float(x) for x in np.percentile(estimates, [2.5, 97.5]))


def template_sign_flip(pairs, query_meta):
    by_template = defaultdict(list)
    for query_id, ratio in pairs:
        by_template[query_meta[query_id]["template_id"]].append(math.log(max(ratio, 1e-12)))
    values = [float(np.mean(logs)) for _, logs in sorted(by_template.items())]
    count = len(values)
    if not count:
        return float("nan"), 0.0
    observed = abs(sum(values) / count)
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=count):
        statistic = abs(sum(value * sign for value, sign in zip(values, signs)) / count)
        if statistic >= observed - 1e-12:
            extreme += 1
    return extreme / (2 ** count), 1 / (2 ** max(count - 1, 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    rows = load_rows(args.raw)
    measured_rows = [r for r in rows if r.get("phase") == "measured" and r.get("plan_fingerprint")]
    if not measured_rows:
        raise SystemExit("No measured rows found in raw.csv")

    metadata_path = args.raw.parent / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        metadata = {}
    note = metadata.get("note", "")

    query_meta = {}
    by_query_arm = defaultdict(dict)
    default_runs = defaultdict(dict)
    for row in measured_rows:
        query_id = row["query_id"]
        query_meta[query_id] = {"template_id": row["template_id"], "split": row["split"],
                                "n_tables": int(row["n_tables"]), "table_set": row.get("table_set", "")}
        if row.get("arm") == "default":
            run_key = (row.get("plan_fingerprint"), row.get("repeat"))
            default_runs[query_id].setdefault(run_key, row)
        if row.get("hint_used", "").lower() != "true":
            continue
        key = (query_id, row["arm"], row["plan_fingerprint"], row["repeat"])
        by_query_arm[(query_id, row["arm"])].setdefault(key, float(row["exec_ms"]))
    med = {key: float(np.median(list(vals.values()))) for key, vals in by_query_arm.items()}
    default_stats = {}
    for query_id, run_rows in default_runs.items():
        times = [float(row["exec_ms"]) for row in run_rows.values() if row.get("exec_ms")]
        cv = float(np.std(times, ddof=1) / np.mean(times)) if len(times) > 1 and np.mean(times) else None
        buffer_rows = [row for row in run_rows.values() if row.get("buffer_parser_version") == "2"
                       and row.get("shared_hit_blocks", "") != "" and row.get("shared_read_blocks", "") != ""]
        hits = sum(int(row.get("shared_hit_blocks") or 0) for row in buffer_rows)
        reads = sum(int(row.get("shared_read_blocks") or 0) for row in buffer_rows)
        default_stats[query_id] = {"median": float(np.median(times)) if times else None,
                                   "cv": cv, "hits": hits, "reads": reads,
                                   "read_share": reads / (hits + reads) if buffer_rows and hits + reads else None,
                                   "buffer_rows": len(buffer_rows), "r": len(times)}
        if query_id in query_meta:
            query_meta[query_id]["cv"] = cv
    noisy_queries = {query_id for query_id, stats in default_stats.items()
                     if stats["cv"] is not None and stats["cv"] > 0.2}
    arms = sorted({arm for _, arm in med if arm not in {"default", "oracle"}})
    dry_path = args.raw.parent / "dry_run.csv"
    dry_rows = []
    if dry_path.exists():
        with dry_path.open(newline="", encoding="utf-8") as f:
            dry_rows = list(csv.DictReader(f))
    default_fps, arm_fps = defaultdict(set), defaultdict(set)
    for row in dry_rows:
        fp = row.get("plan_fingerprint", "")
        if not fp:
            continue
        if row.get("arm") == "default":
            default_fps[row["query_id"]].add(fp)
        elif row.get("arm") != "query_validation" and row.get("hint_used", "").lower() == "true":
            arm_fps[(row["query_id"], row["arm"])].add(fp)
    if not default_fps:
        for (query_id, arm), vals in by_query_arm.items():
            if arm == "default":
                default_fps[query_id].update(key[2] for key in vals)
    if not arm_fps:
        for (query_id, arm), vals in by_query_arm.items():
            if arm != "default":
                arm_fps[(query_id, arm)].update(key[2] for key in vals)

    same_only, differs = {}, {}
    all_pairs_by_arm = {}
    for arm in arms:
        pairs = []
        for query_id in query_meta:
            if (query_id, "default") not in med or (query_id, arm) not in med:
                continue
            fps = arm_fps.get((query_id, arm), set())
            defaults = default_fps.get(query_id, set())
            changed = bool(fps and defaults and not fps.issubset(defaults))
            differs[(query_id, arm)] = changed
            same_only[(query_id, arm)] = bool(fps and defaults and fps.issubset(defaults))
            ratio = med[(query_id, "default")] / max(med[(query_id, arm)], 1e-12)
            if same_only[(query_id, arm)]:
                ratio = 1.0  # identical fingerprints reuse the same measurements and are ties by construction
            pairs.append((query_id, ratio))
        all_pairs_by_arm[arm] = pairs

    groupers = [
        ("all queries", lambda m: True),
        ("train templates", lambda m: m["split"] == "train"),
        ("test templates", lambda m: m["split"] == "test"),
        ("n_tables=2", lambda m: m["n_tables"] == 2),
        ("n_tables=3", lambda m: m["n_tables"] == 3),
        ("n_tables=4", lambda m: m["n_tables"] == 4),
    ]
    out = ["# Join-order evaluation summary", "", "**Workload note:** Workload parameters were chosen by calibrating the default plan only.", ""]
    out.append(f"**Note:** {note}" if note else "**Note:** (none supplied)")
    out += ["", f"Measured rows with fingerprints: {len(measured_rows)}; paired query-arm medians: {len(med)}.",
            f"Checkpoint train_count: {', '.join(f'{k}={v.get('train_count', 'unknown')}' for k, v in metadata.get('checkpoint', {}).items()) or 'unavailable'}.",
            "All comparisons use hint-used rows only. Same-plan cases are forced ties because measurements are reused by physical plan.",
            "The ±5% tie band is smaller than the noise floor when the median default-plan CV exceeds 0.05.",
            "Shared-buffer summaries ignore raw rows without buffer_parser_version=2; rerun measurements to populate corrected counters.", ""]
    for arm in arms:
        out += [f"## {arm} vs default", "", "Counts and intervals below are reported for all paired queries and for queries where at least one arm fingerprint differs from default.", ""]
        for group_name, predicate in groupers:
            group_all = [(q, ratio) for q, ratio in all_pairs_by_arm[arm] if predicate(query_meta[q])]
            cohorts = [("all paired queries", group_all),
                       ("different-plan queries", [(q, ratio) for q, ratio in group_all if differs.get((q, arm), False)])]
            for cohort_name, pairs in cohorts:
                n_templates = len({query_meta[q]["template_id"] for q, _ in pairs})
                out += [f"### {group_name} -- {cohort_name}", "", f"Queries: {len(pairs)}; templates: {n_templates}.", ""]
                if not pairs:
                    out += ["No paired queries in this cohort.", ""]
                    continue
                ratios = [ratio for _, ratio in pairs]
                logmean = sum(math.log(max(ratio, 1e-12)) for ratio in ratios) / len(ratios)
                geom = math.exp(logmean)
                q_ci = ci_geom(pairs, {q: q for q, _ in pairs})
                template_map = {q: query_meta[q]["template_id"] for q, _ in pairs}
                tpl_ci = ci_geom(pairs, template_map) if n_templates >= 8 else None
                wins = sum(ratio > 1.05 for ratio in ratios)
                losses = sum(ratio < 1 / 1.05 for ratio in ratios)
                ties = len(ratios) - wins - losses
                noise_wins = noise_losses = 0
                for query_id, ratio in pairs:
                    cv = default_stats.get(query_id, {}).get("cv")
                    tie_band = max(0.05, 2 * cv) if cv is not None else 0.05
                    if ratio > 1 + tie_band:
                        noise_wins += 1
                    elif ratio < 1 / (1 + tie_band):
                        noise_losses += 1
                noise_ties = len(ratios) - noise_wins - noise_losses
                same_count = sum(1 for q, _ in pairs if same_only.get((q, arm), False))
                paired_qids = {q for q, _ in pairs}
                same_variant_count = sum(1 for r in dry_rows if r.get("arm") == arm and r.get("query_id") in paired_qids
                                         and r.get("plan_fingerprint") and r["plan_fingerprint"] in default_fps.get(r.get("query_id"), set()))
                variant_count = sum(1 for r in dry_rows if r.get("arm") == arm and r.get("query_id") in paired_qids and r.get("plan_fingerprint"))
                worst_slowdown = max([1.0] + [1 / ratio for ratio in ratios])
                arm_times = [med[(q, arm)] for q, _ in pairs]
                stat_rows = [r for r in rows if r.get("arm") == arm and r.get("query_id") in {q for q, _ in pairs}
                             and r.get("timeout", "").lower() == "true"]
                timeout_keys = {(r.get("query_id"), r.get("plan_fingerprint"), r.get("repeat")) for r in stat_rows}
                skipped = sum(1 for r in stat_rows if r.get("phase") == "skipped_after_warmup_timeout")
                try:
                    nonzero_logs = [math.log(ratio) for ratio in ratios if not math.isclose(ratio, 1.0, rel_tol=0, abs_tol=1e-12)]
                    pvalue = float(wilcoxon(nonzero_logs, zero_method="wilcox").pvalue) if nonzero_logs else 1.0
                except ValueError:
                    pvalue = 1.0
                flip_p, min_p = template_sign_flip(pairs, query_meta)
                out += ["| Statistic | Value | n (queries/templates) |", "|---|---:|---:|",
                        f"| Geometric mean speedup (default / arm) | {geom:.4f} | {len(pairs)}/{n_templates} |",
                        f"| Query bootstrap 95% CI (10,000 draws; main CI) | [{q_ci[0]:.4f}, {q_ci[1]:.4f}] | {len(pairs)}/{n_templates} |",
                        f"| Template-cluster bootstrap 95% CI | {f'[{tpl_ci[0]:.4f}, {tpl_ci[1]:.4f}]' if tpl_ci else 'n/a (fewer than 8 templates)'} | {len(pairs)}/{n_templates} |",
                        f"| Win / tie / loss (+/-5% band only) | {wins} / {ties} / {losses} | {len(pairs)}/{n_templates} |",
                        f"| Win / tie / loss (noise-aware max(5%, 2x default CV)) | {noise_wins} / {noise_ties} / {noise_losses} | {len(pairs)}/{n_templates} |",
                        f"| Same-plan query pairs tied by construction | {same_count} | {len(pairs)}/{n_templates} |",
                        f"| Same-plan hint variants tied by construction | {same_variant_count}/{variant_count} | {len(pairs)}/{n_templates} |",
                        f"| Worst single-query slowdown | {worst_slowdown:.4f}x | {len(pairs)}/{n_templates} |",
                        f"| Timeout outcomes (including skipped repeats) | {len(timeout_keys)}; skipped={skipped} | {len(pairs)}/{n_templates} |",
                        f"| Total time (sum of per-query arm medians) | {sum(arm_times):.3f} ms | {len(pairs)}/{n_templates} |",
                        f"| Median / P95 / P99 arm time | {np.median(arm_times):.3f} / {np.percentile(arm_times,95):.3f} / {np.percentile(arm_times,99):.3f} ms | {len(pairs)}/{n_templates} |",
                        f"| Query Wilcoxon signed-rank p-value on log ratios | {pvalue:.6g}; zero log-ratios discarded (zero_method='wilcox') | {len(pairs)}/{n_templates} |",
                        f"| Exact template sign-flip p-value | {flip_p:.6g}; minimum attainable={min_p:.6g} | {len(pairs)}/{n_templates} |", ""]

        out += [f"### {arm} template-level results", "", "| Template | Table set | Split | Queries | Geometric mean speedup |", "|---|---|---|---:|---:|"]
        for template in get_templates(bool(metadata.get("extra_unseen"))):
            table_set = "{" + ",".join(template["tables"]) + "}"
            template_pairs = [(q, ratio) for q, ratio in all_pairs_by_arm[arm] if query_meta[q]["template_id"] == template["id"]]
            speedup = math.exp(sum(math.log(max(ratio, 1e-12)) for _, ratio in template_pairs) / len(template_pairs)) if template_pairs else None
            out.append(f"| {template['id']} | {table_set} | {template['split']} | {len(template_pairs)} | {speedup:.4f} |" if speedup is not None else f"| {template['id']} | {table_set} | {template['split']} | 0 | n/a |")
        out.append("")

    default_cvs = [stats["cv"] for stats in default_stats.values() if stats["cv"] is not None]
    median_cv = float(np.median(default_cvs)) if default_cvs else float("nan")
    low_repeat = sum(1 for stats in default_stats.values() if stats["r"] < 4)
    repeat_reliability = f"CV unreliable (R<4): {low_repeat}/{len(default_stats)} default-plan queries have fewer than four measured repeats." if low_repeat else "All default-plan CV values have at least four measured repeats."
    out += ["## Noise and per-query default-plan measurements", "",
            f"Noisy queries (default-plan CV > 0.2): {len(noisy_queries)}/{len(default_stats)}. Median CV: {median_cv:.4f} across {len(default_cvs)} queries.",
            repeat_reliability,
            "Existing arm-vs-default tables above include all measured queries. This table repeats the geometric speedup after excluding noisy queries.", "",
            "| Arm | n all | n after noisy excluded | Speedup all | Speedup excluding noisy | Clean query-bootstrap 95% CI |", "|---|---:|---:|---:|---:|---:|"]
    for arm in arms:
        pairs = all_pairs_by_arm[arm]
        clean_pairs = [(q, ratio) for q, ratio in pairs if q not in noisy_queries]
        all_geom = math.exp(sum(math.log(max(ratio, 1e-12)) for _, ratio in pairs) / len(pairs)) if pairs else float("nan")
        clean_geom = math.exp(sum(math.log(max(ratio, 1e-12)) for _, ratio in clean_pairs) / len(clean_pairs)) if clean_pairs else float("nan")
        clean_ci = ci_geom(clean_pairs, {q: q for q, _ in clean_pairs}) if clean_pairs else (float("nan"), float("nan"))
        out.append(f"| {arm} | {len(pairs)} | {len(clean_pairs)} | {all_geom:.4f} | {clean_geom:.4f} | [{clean_ci[0]:.4f}, {clean_ci[1]:.4f}] |")
    out += ["", "| Query | Default median (ms) | R | Default CV | Shared-read share | Noisy? |", "|---|---:|---:|---:|---:|---|"]
    for query_id in sorted(default_stats):
        stats = default_stats[query_id]
        median_text = f"{stats['median']:.3f}" if stats["median"] is not None else "n/a"
        cv_text = f"{stats['cv']:.4f}" if stats["cv"] is not None else "n/a"
        read_text = f"{stats['read_share']:.1%}" if stats["read_share"] is not None else "n/a"
        out.append(f"| {query_id} | {median_text} | {stats['r']} | {cv_text} | {read_text} | {'yes' if query_id in noisy_queries else 'no'} |")
    if median_cv > 0.05:
        out.append(f"\nObserved median CV is {median_cv:.4f}; the ±5% tie band is smaller than this noise floor.")
    else:
        out.append(f"\nObserved median CV is {median_cv:.4f}; it does not exceed 0.05, so the tie-band/noise-floor condition was not triggered.")

    hint_orders = {}
    for row in dry_rows:
        if row.get("arm") in {"agent_greedy", "heur_large"} and row.get("hint_used", "").lower() == "true":
            hint_orders.setdefault((row.get("query_id"), row.get("arm")), row.get("order", ""))
    compared_qids = sorted(q for q in query_meta
                           if (q, "agent_greedy") in med and (q, "heur_large") in med)
    identical = [(q, hint_orders.get((q, "agent_greedy"), ""), hint_orders.get((q, "heur_large"), ""))
                 for q in compared_qids]
    identical = [row for row in identical if row[1] and row[2]]
    same_order_count = sum(1 for _, agent_order, large_order in identical if agent_order == large_order)
    agent_large_pairs = []
    for query_id in compared_qids:
        ratio = med[(query_id, "heur_large")] / max(med[(query_id, "agent_greedy")], 1e-12)
        agent_fps = arm_fps.get((query_id, "agent_greedy"), set())
        large_fps = arm_fps.get((query_id, "heur_large"), set())
        if agent_fps and large_fps and agent_fps == large_fps:
            ratio = 1.0
        agent_large_pairs.append((query_id, ratio))
    clean_agent_large = [(q, ratio) for q, ratio in agent_large_pairs if q not in noisy_queries]
    out += ["", "## agent_greedy vs heur_large", "",
            f"Identical-order share in measured quick queries: {same_order_count}/{len(identical)} ({same_order_count/max(1, len(identical)):.1%}).",
            "Speedup is heur_large median / agent_greedy median, so values above 1 favor agent_greedy. Fingerprint-identical pairs are tied by construction.", "",
            "| Cohort | Queries | Geometric mean speedup | Query-bootstrap 95% CI | W/T/L ±5% | W/T/L noise-aware |", "|---|---:|---:|---:|---:|---:|"]
    for cohort, pairs in (("all", agent_large_pairs), ("excluding noisy", clean_agent_large)):
        if pairs:
            geom = math.exp(sum(math.log(max(ratio, 1e-12)) for _, ratio in pairs) / len(pairs))
            ci = ci_geom(pairs, {q: q for q, _ in pairs})
            wins = sum(ratio > 1.05 for _, ratio in pairs)
            losses = sum(ratio < 1 / 1.05 for _, ratio in pairs)
            ties = len(pairs) - wins - losses
            noise_wins = noise_losses = 0
            for query_id, ratio in pairs:
                cv = default_stats.get(query_id, {}).get("cv")
                tie_band = max(0.05, 2 * cv) if cv is not None else 0.05
                if ratio > 1 + tie_band:
                    noise_wins += 1
                elif ratio < 1 / (1 + tie_band):
                    noise_losses += 1
            noise_ties = len(pairs) - noise_wins - noise_losses
            out.append(f"| {cohort} | {len(pairs)} | {geom:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {wins} / {ties} / {losses} | {noise_wins} / {noise_ties} / {noise_losses} |")
        else:
            out.append(f"| {cohort} | 0 | n/a | n/a | n/a | n/a |")

    total_measured = len(measured_rows)
    candidate_rows = [r for r in dry_rows if r.get("arm") not in {"default", "query_validation"}]
    unused = sum(1 for r in candidate_rows if r.get("hint_used", "").lower() != "true")
    changed_candidates = 0
    for row in candidate_rows:
        q, arm, fp = row.get("query_id"), row.get("arm"), row.get("plan_fingerprint")
        if fp and default_fps.get(q) and fp not in default_fps[q]:
            changed_candidates += 1
    agent_hints = {r.get("order", "") for r in dry_rows if r.get("arm", "").startswith("agent_") and r.get("order")}
    out += ["## Diagnostics", "", f"- Candidate hint plans with a different fingerprint from default: {changed_candidates}/{len(candidate_rows)} ({changed_candidates/max(1,len(candidate_rows)):.1%}); n={len(candidate_rows)} hint-query variants across {len({r.get('template_id') for r in candidate_rows})} templates.",
            f"- Distinct trained/untrained agent hints in dry-run: {len(agent_hints)}; the learned policy receives table features but no predicates, so a table set gets the same greedy hint across parameter values.",
            f"- Unused hinted plan variants: {unused}/{len(candidate_rows)} ({unused/max(1,len(candidate_rows)):.1%}); excluded from arm comparisons.",
            f"- Measured raw rows with fingerprints: {total_measured}."]
    out += ["", "| Arm | Plan variants | Different from default | Unused hints | Distinct hint orders |", "|---|---:|---:|---:|---:|"]
    dry_arms = sorted({r.get("arm", "") for r in candidate_rows})
    for arm in dry_arms:
        arm_rows = [r for r in candidate_rows if r.get("arm") == arm]
        diff_count = sum(1 for r in arm_rows if r.get("plan_fingerprint") and r["plan_fingerprint"] not in default_fps.get(r.get("query_id"), set()))
        unused_count = sum(1 for r in arm_rows if r.get("hint_used", "").lower() != "true")
        hints = {r.get("order", "") for r in arm_rows if r.get("order")}
        out.append(f"| {arm} | {len(arm_rows)} | {diff_count}/{len(arm_rows)} ({diff_count/max(1,len(arm_rows)):.1%}) | {unused_count}/{len(arm_rows)} ({unused_count/max(1,len(arm_rows)):.1%}) | {len(hints)} |")

    validation_rows = [row for row in dry_rows if row.get("arm") == "query_validation"]
    dry_query_templates = {row.get("query_id"): row.get("template_id") for row in validation_rows}
    dry_orders = {}
    for row in dry_rows:
        if row.get("arm") in {"agent_greedy", "heur_large", "heur_small"} and row.get("hint_used", "").lower() == "true":
            dry_orders.setdefault((row.get("query_id"), row.get("arm")), row.get("order", ""))
    all_dry_qids = sorted(dry_query_templates)
    order_comparisons = {}
    for baseline in ("heur_large", "heur_small"):
        comparable = [q for q in all_dry_qids if dry_orders.get((q, "agent_greedy")) and dry_orders.get((q, baseline))]
        equal_count = sum(dry_orders[(q, "agent_greedy")] == dry_orders[(q, baseline)] for q in comparable)
        order_comparisons[baseline] = (equal_count, len(comparable))
    out += ["", "## Dry-run order agreement (all query instances)", "",
            f"Coverage: {len(all_dry_qids)} query instances across {len(set(dry_query_templates.values()))} templates in dry_run.csv; order counts use hint-used rows only.",
            f"agent_greedy vs heur_large: {order_comparisons['heur_large'][0]}/{order_comparisons['heur_large'][1]} ({order_comparisons['heur_large'][0]/max(1, order_comparisons['heur_large'][1]):.1%}) identical orders.",
            f"agent_greedy vs heur_small: {order_comparisons['heur_small'][0]}/{order_comparisons['heur_small'][1]} ({order_comparisons['heur_small'][0]/max(1, order_comparisons['heur_small'][1]):.1%}) identical orders.",
            "`heur_large` is largest-table-first using each table's single-table EXPLAIN row estimate.", "",
            "| Template | Table set | Query instances | Equals largest-table-first? | Equal to heur_large | Equal to heur_small |", "|---|---|---:|---|---:|---:|"]
    template_by_id = {template["id"]: template for template in get_templates(True)}
    for template_id in sorted(set(dry_query_templates.values())):
        qids = sorted(q for q, tid in dry_query_templates.items() if tid == template_id)
        template = template_by_id.get(template_id, {})
        table_set = "{" + ",".join(template.get("tables", ())) + "}"
        equalities = {}
        for baseline in ("heur_large", "heur_small"):
            comparable = [q for q in qids if dry_orders.get((q, "agent_greedy")) and dry_orders.get((q, baseline))]
            equal_count = sum(dry_orders[(q, "agent_greedy")] == dry_orders[(q, baseline)] for q in comparable)
            equalities[baseline] = (equal_count, len(comparable))
        large_count, large_n = equalities["heur_large"]
        large_flag = "yes" if large_n and large_count == large_n else ("no" if large_n else "n/a")
        out.append(f"| {template_id} | {table_set} | {len(qids)} | {large_flag} | {large_count}/{large_n} | {equalities['heur_small'][0]}/{equalities['heur_small'][1]} |")
    out.append(f"- Default run-to-run CV median: {median_cv:.4f} across {len(default_cvs)} queries/{len({query_meta[q]['template_id'] for q in default_stats if q in query_meta})} templates.")

    oracle_plans = defaultdict(list)
    for r in measured_rows:
        if r.get("arm") == "oracle" and r.get("hint_used", "").lower() == "true":
            oracle_plans[(r["query_id"], r["plan_fingerprint"])].append(float(r["exec_ms"]))
    oracle = defaultdict(list)
    for (query_id, _), values in oracle_plans.items():
        oracle[query_id].append(float(np.median(values)))
    regrets = []
    for (query_id, arm), value in med.items():
        if arm.startswith("agent_") and oracle.get(query_id):
            best = min(oracle[query_id])
            regrets.append((value / best - 1, query_id))
    out.append(f"- Agent regret versus oracle: {np.mean([r for r, _ in regrets]):.4f} across {len(regrets)} query-arm pairs/{len({query_meta[q]['template_id'] for _, q in regrets})} templates." if regrets else "- Agent regret versus oracle: n/a (no paired oracle measurements).")

    oracle_orders = defaultdict(set)
    for row in dry_rows:
        if row.get("arm") == "oracle" and row.get("hint_used", "").lower() == "true" and row.get("plan_fingerprint"):
            oracle_orders[(row.get("query_id"), row["plan_fingerprint"])].add(row.get("order", ""))
    oracle_runs = defaultdict(dict)
    for row in measured_rows:
        if row.get("arm") == "oracle" and row.get("hint_used", "").lower() == "true":
            oracle_runs[(row["query_id"], row["plan_fingerprint"])].setdefault(row.get("repeat"), float(row["exec_ms"]))
            if row.get("order"):
                oracle_orders[(row["query_id"], row["plan_fingerprint"])].add(row["order"])
    oracle_plan_medians = {key: float(np.median(list(values.values()))) for key, values in oracle_runs.items()}
    order_times = defaultdict(list)
    for (query_id, fingerprint), plan_time in oracle_plan_medians.items():
        for order in oracle_orders.get((query_id, fingerprint), set()):
            if order:
                order_times[(query_id, order)].append(plan_time)
    order_times = {key: float(np.median(values)) for key, values in order_times.items()}
    query_oracle = {}
    for query_id in query_meta:
        choices = [(time_ms, order) for (qid, order), time_ms in order_times.items() if qid == query_id]
        if choices and (query_id, "default") in med:
            best_time, best_order = min(choices)
            query_oracle[query_id] = {"order": best_order, "time": best_time,
                                      "speedup": med[(query_id, "default")] / max(best_time, 1e-12)}

    out += ["", "## Static-order ceiling for a policy that cannot see predicates", "",
            "For each template, the best static order is the order with the lowest geometric-mean execution time across that template's parameter values. The overall speedup uses each template's best static order across all eligible parameter queries.", ""]
    dry_qids_by_template = defaultdict(list)
    for query_id, template_id in dry_query_templates.items():
        dry_qids_by_template[template_id].append(query_id)
    template_static = {}
    static_query_speedups = []
    static_rows = []
    if not query_oracle:
        out += ["No measured oracle rows are available yet; the static-order ceiling and per-query oracle speedups will appear after an `--oracle` measurement run.", ""]
    else:
        out += ["| Template | Parameter queries | Best static order | Static geometric-mean (ms) | Default geometric-mean (ms) | Speedup vs default |", "|---|---:|---|---:|---:|---:|"]
        for template_id in sorted(dry_qids_by_template):
            qids = sorted(q for q in dry_qids_by_template[template_id] if q in query_oracle and (q, "default") in med)
            all_template_qids = dry_qids_by_template[template_id]
            if not qids or len(qids) != len(all_template_qids):
                static_rows.append(f"| {template_id} | {len(qids)}/{len(all_template_qids)} | n/a | n/a | n/a | n/a |")
                continue
            common_orders = set.intersection(*(set(order for (qid, order) in order_times if qid == q) for q in qids)) if qids else set()
            order_geomeans = []
            for order in common_orders:
                values = [order_times[(q, order)] for q in qids]
                gmean = math.exp(sum(math.log(max(value, 1e-12)) for value in values) / len(values))
                order_geomeans.append((gmean, order))
            if not order_geomeans:
                static_rows.append(f"| {template_id} | {len(qids)}/{len(all_template_qids)} | n/a (no common measured order) | n/a | n/a | n/a |")
                continue
            static_time, best_order = min(order_geomeans)
            default_time = math.exp(sum(math.log(max(med[(q, "default")], 1e-12)) for q in qids) / len(qids))
            speedup = default_time / max(static_time, 1e-12)
            template_static[template_id] = {"order": best_order, "static_time": static_time, "default_time": default_time,
                                            "speedup": speedup, "query_ids": qids}
            static_query_speedups.extend((q, med[(q, "default")] / max(order_times[(q, best_order)], 1e-12)) for q in qids)
            static_rows.append(f"| {template_id} | {len(qids)}/{len(all_template_qids)} | {best_order} | {static_time:.3f} | {default_time:.3f} | {speedup:.4f} |")
        out.extend(static_rows)
        if static_query_speedups:
            overall = math.exp(sum(math.log(max(speedup, 1e-12)) for _, speedup in static_query_speedups) / len(static_query_speedups))
            out.append(f"\nOverall ceiling speedup vs default: {overall:.4f} over {len(static_query_speedups)} parameter queries in {len(template_static)} complete templates.")
        out += ["", "### Per-query oracle speedup", "", "Speedup is default median / fastest measured oracle order; this oracle can vary its order by parameter value.", "",
                "| Query | Best oracle order | Default median (ms) | Oracle best (ms) | Speedup |", "|---|---|---:|---:|---:|"]
        for query_id in sorted(query_oracle):
            best = query_oracle[query_id]
            out.append(f"| {query_id} | {best['order']} | {med[(query_id, 'default')]:.3f} | {best['time']:.3f} | {best['speedup']:.4f} |")
        oracle_gmean = math.exp(sum(math.log(max(item["speedup"], 1e-12)) for item in query_oracle.values()) / len(query_oracle))
        out.append(f"\nOverall per-query oracle geometric-mean speedup: {oracle_gmean:.4f} across {len(query_oracle)} queries.")

    out += ["", "## Agent decision table", "", "Hint order is from dry-run; plan difference compares its fingerprint with the no-hint default.",
            "", "| Template | Table set | Split | agent_greedy hint / differs? | agent_untrained_greedy hint / differs? |", "|---|---|---|---|---|"]
    for template in get_templates(bool(metadata.get("extra_unseen"))):
        decisions = []
        for arm in ("agent_greedy", "agent_untrained_greedy"):
            candidates = [r for r in dry_rows if r.get("template_id") == template["id"] and r.get("arm") == arm]
            chosen = candidates[0] if candidates else None
            if chosen:
                default = default_fps.get(chosen["query_id"], set())
                differs_text = "yes" if chosen.get("plan_fingerprint") and chosen["plan_fingerprint"] not in default else "no"
                decisions.append(f"{chosen.get('order') or '(no hint)'} / {differs_text}")
            else:
                decisions.append("n/a / n/a")
        table_set = "{" + ",".join(template["tables"]) + "}"
        out.append(f"| {template['id']} | {table_set} | {template['split']} | {decisions[0]} | {decisions[1]} |")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
