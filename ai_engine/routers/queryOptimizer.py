"""
queryOptimizer.py
-----------------
Layer 1: PPO-based join order optimization.

Bug fixes applied
-----------------
1. Alias handling    : `state.tables` carries aliases (pg_hint_plan identifiers);
                       `state.real_names` carries physical names for schema stats.
2. Hint direction    : Leading hint uses nested-parenthesis left-deep form
                       Leading((((a b) c) d)) instead of flat Leading(a b c).
3. PPO execution     : log_prob_old is stored server-side and never exposed in the
                       HTTP response, so the client cannot tamper with it.
4. Stochastic policy : join order is SAMPLED from a Categorical distribution over
                       attention weights, not taken as the argmax.  The log-prob of
                       the full permutation sequence is computed and stored.
5. Disconnected graph: join_conditions (parsed ON-clause edges from Java) are
                       threaded through sample_action() so the sampler only places
                       tables connected to the current prefix — no cross-products.
6. Alias injection   : aliases_safe=False (set by Java whitelist check) forces the
                       hint to be omitted entirely; original SQL runs unhinted.
"""

import re
from fastapi import APIRouter
from models.model import QueryState
from agent.trainer_instance import trainer
from agent.experience_buffer import store_pending
from vectorizer.schema_vectorizer import build_state_tensor
from analytics.analytics_store import new_record

router   = APIRouter()
ai_model = trainer.model

# Same whitelist as QueryController.java — validated again server-side for
# defence in depth (the Java value could be misconfigured).
_SAFE_ALIAS = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def _aliases_safe(aliases: list[str]) -> bool:
    return all(_SAFE_ALIAS.match(a) for a in aliases)


def _nested_leading(order: list[str]) -> str:
    """
    Build the nested-parenthesis Leading hint for a left-deep join tree.

    order = [a, b, c, d]  →  Leading((((a b) c) d))

    pg_hint_plan treats nested pairs as "join the inner group first",
    so this fully pins the join tree direction — unlike flat Leading(a b c)
    which leaves the planner free to reorder.
    """
    if len(order) == 1:
        return order[0]
    result = f"({order[0]} {order[1]})"
    for table in order[2:]:
        result = f"({result} {table})"
    return result


@router.post("/optimize")
async def optimize_query(state: QueryState):
    # aliases    → identifiers pg_hint_plan matches in the SQL
    # stat_names → physical table names for schema statistics
    aliases         = state.tables
    stat_names      = state.real_names if state.real_names else state.tables
    join_conditions = state.join_conditions   # list of "leftAlias,rightAlias" strings

    print(f"[Optimize] Aliases={aliases}  StatNames={stat_names}  "
          f"Edges={join_conditions}  AliasesSafe={state.aliases_safe}")

    # 1. Build state tensor using REAL table names (accurate schema stats)
    state_tensor = build_state_tensor(stat_names)

    # 2. Graph-aware stochastic sampling.
    #    sample_action() respects join_conditions so every step in the sampled
    #    order is connected to the already-placed prefix — no cross-products.
    optimized_order, log_prob_old = trainer.sample_action(
        state_tensor, aliases, join_conditions)

    # 3. XAI explanation (no gradient needed)
    import torch
    with torch.no_grad():
        _, attention_weights = ai_model(state_tensor)
    xai_explanation = {
        alias: round(attention_weights[i].item() * 100, 2)
        for i, alias in enumerate(aliases)
    }

    # 4. Build hint SQL — or fall back to the original SQL when aliases are
    #    unsafe (Bug 6: comment-injection guard, defence-in-depth layer).
    #    Java already set aliases_safe=False and logged the offending aliases;
    #    we honour that flag here regardless of _aliases_safe() result so a
    #    misconfigured Java side cannot bypass the check.
    hint_allowed = state.aliases_safe and _aliases_safe(aliases)
    if hint_allowed and len(optimized_order) > 1:
        leading_expr = _nested_leading(optimized_order)
        final_sql    = f"/*+ Leading({leading_expr}) */ {state.original_sql}"
    else:
        # Unsafe aliases or single-table query → run without hint
        final_sql = state.original_sql
        if not hint_allowed:
            print("[SECURITY] Hint suppressed — unsafe alias(es) detected.")

    # 5. Create pending analytics record (outcome filled by /feedback)
    rec = new_record(
        tables        = aliases,
        original_sql  = state.original_sql,
        chosen_order  = optimized_order,
        is_anomalous  = False,
        anomaly_score = 0.0,
    )

    # 6. Store log_prob_old SERVER-SIDE — never sent to client.
    store_pending(
        query_id        = rec.query_id,
        state_tensor    = state_tensor,
        log_prob_old    = log_prob_old,
        aliases         = aliases,
        stat_names      = stat_names,
        join_conditions = join_conditions,
    )

    return {
        "status":          "success",
        "optimized_query": final_sql,
        "chosen_order":    optimized_order,
        "xai_explanation": xai_explanation,
        "query_id":        rec.query_id,
        "hint_applied":    hint_allowed and len(optimized_order) > 1,
        # log_prob_old intentionally NOT returned — stored server-side only
    }