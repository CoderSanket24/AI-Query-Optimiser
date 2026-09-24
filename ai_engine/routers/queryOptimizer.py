"""
queryOptimizer.py
-----------------
Layer 1: PPO-based join order optimization.

The old Isolation Forest pre-query gate (Part 5 v1) is REMOVED.
Anomaly detection now happens AFTER execution in /feedback using
(wait_time_ms, exec_time_ms, active_connections) — the correct DoS signal.

This router only optimises the join order and returns the hinted SQL.
"""

from fastapi import APIRouter
from models.model import QueryState
from agent.trainer_instance import trainer
from agent.experience_buffer import store_pending
from vectorizer.schema_vectorizer import build_state_tensor
from analytics.analytics_store import new_record

router   = APIRouter()
ai_model = trainer.model


@router.post("/optimize")
async def optimize_query(state: QueryState):
    print(f"[Optimize] Tables: {state.tables}")

    # 1. Build real state tensor from live schema stats
    state_tensor = build_state_tensor(state.tables)

    # 2. PPO forward pass — produce attention weights and join order
    _, attention_weights = ai_model(state_tensor)

    xai_explanation = {
        table: round(attention_weights[i].item() * 100, 2)
        for i, table in enumerate(state.tables)
    }
    optimized_order = [t for t, _ in sorted(
        xai_explanation.items(), key=lambda x: x[1], reverse=True)]
    final_sql = f"/*+ Leading({' '.join(optimized_order)}) */ {state.original_sql}"

    # 3. Compute log_prob_old for PPO ratio (Part 7 — true PPO clipping)
    log_prob_old, first_idx = trainer.compute_action_log_prob(
        state_tensor, state.tables, optimized_order)

    # 4. Create pending analytics record (outcome filled by /feedback)
    rec = new_record(
        tables        = state.tables,
        original_sql  = state.original_sql,
        chosen_order  = optimized_order,
        is_anomalous  = False,   # anomaly detection moved to /feedback
        anomaly_score = 0.0,
    )

    # 5. Store state_tensor + log_prob_old for train_batch() (Part 7)
    store_pending(
        query_id     = rec.query_id,
        state_tensor = state_tensor,
        log_prob_old = log_prob_old,
        first_idx    = first_idx,
        tables       = state.tables,
    )

    return {
        "status":          "success",
        "optimized_query": final_sql,
        "choosen_order":   optimized_order,
        "xai_explanation": xai_explanation,
        "query_id":        rec.query_id,
        "log_prob_old":    round(log_prob_old, 6),
    }