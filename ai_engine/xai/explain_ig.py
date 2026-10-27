from __future__ import annotations

import sys
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE_DIR))

import argparse
import io
import itertools
import random
import traceback
from contextlib import redirect_stderr, redirect_stdout

XAI_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = ENGINE_DIR / "checkpoints" / "ppo_checkpoint.pt"
SAMPLE_STATE_PATH = XAI_DIR / "sample_state.pt"
REPORT_PATH = XAI_DIR / "ig_report.md"

SEED = 20261026
INPUT_DIM = 11
HIDDEN_DIM = 32
REAL_NAMES = ["title", "cast_info", "name"]
ALIASES = ["t", "ci", "n"]
JOIN_CONDITIONS = ["t,ci", "ci,n"]
ALL_TABLES = [
    "title", "cast_info", "movie_keyword", "movie_companies",
    "name", "keyword", "company_name",
]
FEATURE_NAMES = [
    "log_row_count",
    "log_page_count",
    "log_col_count",
    "log_index_count",
    "has_primary_key",
    "has_foreign_key",
    "log_table_size_bytes",
    "avg_row_width_norm",
    "row_density",
    "is_in_query",
    "row_growth_rate",
]
EXPECTED_VALID_ORDERS = {
    ("t", "ci", "n"),
    ("ci", "t", "n"),
    ("ci", "n", "t"),
    ("n", "ci", "t"),
}


def set_seeds(torch, np) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def order_probability_factors(model, state, aliases, order, join_conditions):
    import torch

    adjacency = {alias: set() for alias in aliases}
    for edge in join_conditions:
        left, right = (part.strip() for part in edge.split(",", 1))
        adjacency[left].add(right)
        adjacency[right].add(left)

    with torch.no_grad():
        _, weights = model(state)
    alias_index = {alias: i for i, alias in enumerate(aliases)}
    remaining = list(aliases)
    placed = set()
    factors = []
    for chosen in order:
        if not placed:
            candidates = remaining
        else:
            candidates = [name for name in remaining if adjacency[name] & placed]
            if not candidates:
                candidates = remaining
        candidate_weights = torch.stack(
            [weights[alias_index[name]] for name in candidates]
        )
        candidate_weights = candidate_weights / candidate_weights.sum()
        factors.append((chosen, float(candidate_weights[candidates.index(chosen)])))
        placed.add(chosen)
        remaining.remove(chosen)
    return factors


def format_order(order) -> str:
    return " → ".join(order)


def report_feature_variances(table_state, torch):
    variances = table_state.var(dim=0, unbiased=False)
    variance_values = [float(value) for value in variances]
    varying_indices = [i for i, value in enumerate(variance_values) if value > 1e-12]
    constant_indices = [i for i, value in enumerate(variance_values) if value <= 1e-12]

    print("## Feature variance across the seven IMDb tables")
    print("| Feature | Population variance |")
    print("|---|---:|")
    for feature, variance in zip(FEATURE_NAMES, variance_values):
        print(f"| {feature} | {variance:.12g} |")
    constant_names = [FEATURE_NAMES[i] for i in constant_indices]
    print("\nObserved constant features: " + (", ".join(constant_names) or "none"))
    print(
        "Expected constants to check: has_primary_key, has_foreign_key, "
        "log_index_count (index_count), is_in_query, row_growth_rate."
    )
    print(
        "With the mean baseline, a constant feature has attribution 0 by "
        "construction. With the zero baseline, it can show nonzero attribution "
        "purely because its input differs from 0."
    )
    return variance_values, varying_indices


def print_attribution_table(attributions, table_names, aliases, np):
    values = attributions.detach().cpu().numpy()
    table_labels = [f"{name} ({alias})" for name, alias in zip(table_names, aliases)]

    print("| Table | " + " | ".join(FEATURE_NAMES) + " |")
    print("|---|" + "---:|" * len(FEATURE_NAMES))
    for label, row in zip(table_labels, values):
        print("| " + label + " | " + " | ".join(f"{float(v):.6f}" for v in row) + " |")

    table_totals = np.abs(values).sum(axis=1)
    feature_totals = np.abs(values).sum(axis=0)
    print("\nPer-table totals of |attribution|:")
    for label, total in zip(table_labels, table_totals):
        print(f"- {label}: {total:.8f}")
    print("\nPer-feature totals of |attribution|:")
    for feature, total in zip(FEATURE_NAMES, feature_totals):
        print(f"- {feature}: {total:.8f}")


def score_masked(f, input_state, baseline, positions, torch):
    masked = input_state.detach().clone()
    for table_index, feature_index in positions:
        masked[0, table_index, feature_index] = baseline[0, table_index, feature_index]
    with torch.no_grad():
        return float(f(masked)[0])


def faithfulness_rows(f, input_state, baseline, attributions, positions, seed, torch):
    values = attributions.detach().cpu().numpy()[0]
    ranked_positions = sorted(
        positions,
        key=lambda position: (-abs(float(values[position[0], position[1]])), position),
    )
    baseline_value = float(f(input_state)[0].detach())
    rng = random.Random(seed)
    rows = []
    for k in (1, 2, 3):
        top_positions = ranked_positions[:k]
        top_drop = baseline_value - score_masked(
            f, input_state, baseline, top_positions, torch
        )
        random_drops = []
        for _ in range(20):
            random_positions = rng.sample(positions, k)
            random_drops.append(
                baseline_value - score_masked(
                    f, input_state, baseline, random_positions, torch
                )
            )
        rows.append((k, top_drop, sum(random_drops) / len(random_drops)))
    return rows


def print_faithfulness(f, input_state, baseline, attributions, varying_indices, torch):
    all_positions = [
        (table_index, feature_index)
        for table_index in range(input_state.shape[1])
        for feature_index in range(input_state.shape[2])
    ]
    varying_positions = [
        position for position in all_positions if position[1] in varying_indices
    ]

    print("\n### Faithfulness mini-test")
    print("Drop = log π(original order | x) − log π(order | masked x).")
    print("Each random comparison is the mean over 20 fixed-seed random position sets.")
    for scope, positions in (
        ("ALL positions", all_positions),
        ("VARYING-feature positions", varying_positions),
    ):
        print(f"\n#### Ranked over {scope}")
        print("| k | Top-k drop | Mean random drop (20 sets) |")
        print("|---:|---:|---:|")
        for k, top_drop, random_mean in faithfulness_rows(
            f, input_state, baseline, attributions, positions, SEED, torch
        ):
            print(f"| {k} | {top_drop:.8f} | {random_mean:.8f} |")


def run_demo(args):
    import numpy as np
    import torch
    from captum.attr import IntegratedGradients
    from agent.ppo_agent import ExplainableJoinOptimizer, PPOTrainer

    set_seeds(torch, np)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model = ExplainableJoinOptimizer(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
    trainer = PPOTrainer(model)
    model.load_state_dict(checkpoint["model_state_dict"])
    trainer.train_count = int(checkpoint.get("train_count", 0))
    trainer.baseline = float(checkpoint.get("baseline", 0.0))
    model.eval()

    print(
        f"Checkpoint train_count: {trainer.train_count} | "
        f"baseline: {trainer.baseline:.8f}"
    )
    print("# Integrated Gradients for the PPO join-order policy")
    print(f"Query: SELECT t.title, n.name FROM title t JOIN cast_info ci ON t.id=ci.movie_id JOIN name n ON ci.person_id=n.id WHERE t.production_year=1990 LIMIT 10")
    print(f"Fixed random seed: {SEED}")

    if args.offline:
        bundle = torch.load(SAMPLE_STATE_PATH, map_location="cpu", weights_only=True)
        query_state = bundle["query_state"].to(dtype=torch.float32, device="cpu")
        table_state = bundle["seven_table_state"].to(dtype=torch.float32, device="cpu")
        mean_features = bundle["mean_features"].to(dtype=torch.float32, device="cpu")
        print(f"State source: offline bundle ({SAMPLE_STATE_PATH.name})")
    else:
        from vectorizer.schema_vectorizer import build_state_tensor

        query_state = build_state_tensor(REAL_NAMES)
        table_state = build_state_tensor(ALL_TABLES)
        mean_features = table_state.mean(dim=0)
        torch.save(
            {
                "query_state": query_state.cpu(),
                "seven_table_state": table_state.cpu(),
                "mean_features": mean_features.cpu(),
                "real_names": REAL_NAMES,
                "aliases": ALIASES,
                "join_conditions": JOIN_CONDITIONS,
                "all_table_names": ALL_TABLES,
                "feature_names": FEATURE_NAMES,
            },
            SAMPLE_STATE_PATH,
        )
        print(f"Saved query and seven-table states to: {SAMPLE_STATE_PATH}")
        print("State source: live PostgreSQL via build_state_tensor")

    query_state = query_state.detach().clone().to(dtype=torch.float32, device="cpu")
    table_state = table_state.detach().clone().to(dtype=torch.float32, device="cpu")
    mean_features = mean_features.detach().clone().to(dtype=torch.float32, device="cpu")
    if tuple(query_state.shape) != (len(ALIASES), INPUT_DIM):
        raise ValueError(f"Expected query state shape (3, 11), got {tuple(query_state.shape)}")
    if tuple(table_state.shape) != (len(ALL_TABLES), INPUT_DIM):
        raise ValueError(
            f"Expected seven-table state shape (7, 11), got {tuple(table_state.shape)}"
        )

    print("\n## Seven-table mean feature vector")
    print("| Feature | Mean |")
    print("|---|---:|")
    for feature, value in zip(FEATURE_NAMES, mean_features):
        print(f"| {feature} | {float(value):.8f} |")

    variance_values, varying_indices = report_feature_variances(table_state, torch)
    constant_names = [
        FEATURE_NAMES[index]
        for index, variance in enumerate(variance_values)
        if variance <= 1e-12
    ]
    print("Varying features used by the second faithfulness ranking: "
          + (", ".join(FEATURE_NAMES[i] for i in varying_indices) or "none"))
    print("Constant features detected from these states: "
          + (", ".join(constant_names) or "none"))

    input_state = query_state.unsqueeze(0)

    valid_orders = []
    print("\n## Connected-prefix order probabilities")
    for candidate in itertools.permutations(ALIASES):
        try:
            log_probability = trainer.sequence_log_prob(
                query_state, ALIASES, list(candidate), JOIN_CONDITIONS
            )
            valid_orders.append((candidate, float(torch.exp(log_probability).detach())))
        except Exception as exc:
            print(f"Skipped {format_order(candidate)}: {type(exc).__name__}: {exc}")

    valid_order_set = {order for order, _ in valid_orders}
    if valid_order_set != EXPECTED_VALID_ORDERS:
        raise AssertionError(
            "Expected exactly the four connected orders "
            f"{sorted(EXPECTED_VALID_ORDERS)}, got {sorted(valid_order_set)}"
        )
    for order, probability in valid_orders:
        print(f"- [{format_order(order)}]: {probability:.12f}")
    probability_sum = sum(probability for _, probability in valid_orders)
    print(f"Probability sum: {probability_sum:.12f}")
    if abs(probability_sum - 1.0) > 1e-5:
        raise AssertionError(f"Valid-order probabilities sum to {probability_sum}, not 1.0")
    print("Sum-to-1 check: PASS (tolerance 1e-5)")

    most_probable_order, most_probable_probability = max(
        valid_orders, key=lambda item: item[1]
    )
    current_order = list(most_probable_order)
    print(
        f"Most probable order: {format_order(most_probable_order)} "
        f"(π={most_probable_probability:.12f})"
    )
    print(
        "Policy would sample most often: "
        f"{format_order(most_probable_order)} "
        f"(highest exact sequence probability; π={most_probable_probability:.12f})"
    )
    factors = order_probability_factors(
        model, query_state, ALIASES, most_probable_order, JOIN_CONDITIONS
    )
    print("Order explanation (conditional policy choices): " + " × ".join(
        f"P({alias} next)={probability:.8f}" for alias, probability in factors
    ))

    with torch.no_grad():
        _, attention_weights = model(query_state)
    print("\nAttention weights for the input state (comparison for both baselines):")
    print("| Table | Alias | Attention weight |")
    print("|---|---|---:|")
    for table, alias, weight in zip(REAL_NAMES, ALIASES, attention_weights):
        print(f"| {table} | {alias} | {float(weight):.8f} |")

    def f_for_order(batch):
        scores = []
        for sample in batch:
            scores.append(
                trainer.sequence_log_prob(
                    sample, ALIASES, current_order, JOIN_CONDITIONS
                )
            )
        return torch.stack(scores).reshape(-1)

    # Feature 9 (is_in_query, one-based; index 9 in this list) is always 1.0.
    # With the mean baseline its attribution is zero by construction.
    baselines = [
        ("All-zero baseline", torch.zeros_like(input_state)),
        ("Seven-table mean baseline", mean_features.reshape(1, 1, -1).expand_as(input_state).clone()),
    ]
    input_log_probability = float(f_for_order(input_state)[0].detach())
    print(f"\nExplained order log π(x): {input_log_probability:.12f}")

    for baseline_name, baseline in baselines:
        print(f"\n## {baseline_name}")
        ig = IntegratedGradients(f_for_order)
        steps = 50
        attributions, captum_delta = ig.attribute(
            input_state,
            baselines=baseline,
            n_steps=steps,
            return_convergence_delta=True,
        )
        with torch.no_grad():
            baseline_log_probability = float(f_for_order(baseline)[0])
        attribution_sum = float(attributions.sum().detach())
        output_difference = input_log_probability - baseline_log_probability
        completeness_difference = attribution_sum - output_difference
        if abs(completeness_difference) > 1e-2:
            print(
                f"Completeness difference {completeness_difference:.8f} exceeds "
                "1e-2; retrying Integrated Gradients with n_steps=200."
            )
            steps = 200
            attributions, captum_delta = ig.attribute(
                input_state,
                baselines=baseline,
                n_steps=steps,
                return_convergence_delta=True,
            )
            attribution_sum = float(attributions.sum().detach())
            completeness_difference = attribution_sum - output_difference

        print(f"Integrated Gradients steps used: {steps}")
        print_attribution_table(attributions[0], REAL_NAMES, ALIASES, np)
        print("\nCompleteness check:")
        print(f"- sum(attributions): {attribution_sum:.12f}")
        print(f"- f(x) - f(baseline): {output_difference:.12f}")
        print(f"- difference (sum(attr) - output difference): {completeness_difference:.12f}")
        print(f"- Captum convergence delta: {float(captum_delta[0]):.12f}")
        print(f"- Completeness within 1e-2: {'PASS' if abs(completeness_difference) <= 1e-2 else 'CHECK'}")
        print_faithfulness(
            f_for_order, input_state, baseline, attributions,
            varying_indices, torch,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Standalone Integrated Gradients demo for the JoinSense PPO policy."
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Load the saved xai/sample_state.pt bundle instead of querying PostgreSQL.",
    )
    args = parser.parse_args()

    captured = io.StringIO()
    status = 0
    with redirect_stdout(captured), redirect_stderr(captured):
        try:
            run_demo(args)
        except Exception:
            status = 1
            if "Checkpoint train_count:" not in captured.getvalue():
                print(
                    "Checkpoint train_count: unavailable | baseline: unavailable "
                    "(run failed before checkpoint load)"
                )
            print("\n## Run failure")
            traceback.print_exc(file=sys.stdout)

    report = captured.getvalue()
    report += f"\nFull output saved to `{REPORT_PATH.relative_to(ENGINE_DIR).as_posix()}`.\n"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    sys.stdout.write(report)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
