"""
queryOptimizer.py
-----------------
Layer 1: PPO-based join order optimization.

The old Isolation Forest pre-query gate (Part 5 v1) is REMOVED.
Anomaly detection now happens AFTER execution in /feedback using
(wait_time_ms, active_connections) — the correct DoS signal.

This router only optimises the join order and returns the hinted SQL.

Bug fixes applied
-----------------
1. Alias handling  : `state.tables` carries aliases (pg_hint_plan identifiers);
                     `state.real_names` carries physical names for schema stats.
2. Hint direction  : Leading hint uses nested-parenthesis left-deep form
                     Leading((a (b c))) instead of flat Leading(a b c).
3. PPO execution   : log_prob_old is stored server-side and never exposed in the
                     HTTP response, so the client cannot tamper with it.
4. Stochastic policy: join order is SAMPLED from a Categorical distribution over
                     attention weights, not taken as the argmax.  The log-prob of
                     the full permutation sequence is computed and stored.
"""

from fastapi import APIRouter
from models.model import QueryState
from agent.trainer_instance import trainer
from agent.experience_buffer import store_pending
from vectorizer.schema_vectorizer import build_state_tensor
from analytics.analytics_store import new_record

router   = APIRouter()
ai_model = trainer.model


def _nested_leading(order: list[str]) -> str:
    """
    Build the nested-parenthesis Leading hint for a left-deep join tree.

    For order = [a, b, c, d]:
      Left-deep plan: ((a ⋈ b) ⋈ c) ⋈ d
      Hint:  Leading((((a b) c) d))

    pg_hint_plan interprets nested parens as "join the inner pair first",
    so this fully pins the join tree direction — unlike flat Leading(a b c)
    which only suggests an order but leaves the planner free to reorder.
    """
    if len(order) == 1:
        return order[0]
    # Build from left to right: start with first pair, keep wrapping
    result = f"({order[0]} {order[1]})"
    for table in order[2:]:
        result = f"({result} {table})"
    return result


@router.post("/optimize")
async def optimize_query(state: QueryState):
    # aliases  → identifiers pg_hint_plan matches in the SQL
    # stat_names → physical table names for schema statistics
    aliases    = state.tables
    stat_names = state.real_names if state.real_names else state.tables

    print(f"[Optimize] Aliases: {aliases}  StatNames: {stat_names}")

    # 1. Build state tensor using REAL table names (for accurate schema stats)
    state_tensor = build_state_tensor(stat_names)

    # 2. Sample join order from a stochastic policy (Categorical over attention
    #    weights).  This is required for PPO: the action must come from a
    #    distribution so we can compute a meaningful log-probability ratio.
    #    trainer.sample_action() returns (sampled_order_of_aliases, log_prob).
    optimized_order, log_prob_old = trainer.sample_action(state_tensor, aliases)

    # 3. XAI explanation: re-run forward pass (no_grad) to get weights for display
    import torch
    with torch.no_grad():
        _, attention_weights = ai_model(state_tensor)
    xai_explanation = {
        alias: round(attention_weights[i].item() * 100, 2)
        for i, alias in enumerate(aliases)
    }

    # 4. Build nested-parenthesis Leading hint (pins left-deep join direction)
    leading_expr = _nested_leading(optimized_order)
    final_sql    = f"/*+ Leading({leading_expr}) */ {state.original_sql}"

    # 5. Create pending analytics record (outcome filled by /feedback)
    rec = new_record(
        tables        = aliases,
        original_sql  = state.original_sql,
        chosen_order  = optimized_order,
        is_anomalous  = False,   # anomaly detection moved to /feedback
        anomaly_score = 0.0,
    )

    # 6. Store log_prob_old SERVER-SIDE — never sent to the client.
    #    /feedback will call pop_pending(query_id) to retrieve it, so the
    #    value cannot be tampered with in transit.
    store_pending(
        query_id     = rec.query_id,
        state_tensor = state_tensor,
        log_prob_old = log_prob_old,
        aliases      = aliases,
        stat_names   = stat_names,
    )

    return {
        "status":          "success",
        "optimized_query": final_sql,
        "choosen_order":   optimized_order,
        "xai_explanation": xai_explanation,
        "query_id":        rec.query_id,
        # log_prob_old intentionally NOT returned — stored server-side only
    }