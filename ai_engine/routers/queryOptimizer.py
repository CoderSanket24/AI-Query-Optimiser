from fastapi import APIRouter
from models.model import QueryState
from agent.trainer_instance import trainer
from agent.experience_buffer import store_pending
from vectorizer.schema_vectorizer import build_state_tensor
from anomaly.detector_instance import detector
from anomaly.isolation_forest import get_query_vector
from analytics.analytics_store import new_record

router   = APIRouter()
ai_model = trainer.model


@router.post("/optimize")
async def optimize_query(state: QueryState):
    print(f"[Optimize] Tables: {state.tables}")

    # 1. Build real state tensor from live schema stats
    state_tensor = build_state_tensor(state.tables)

    # 2. Layer 2 — Anomaly detection
    query_vector          = get_query_vector(state_tensor)
    is_anomaly, anm_score = detector.predict(query_vector)

    if is_anomaly:
        print(f"[Anomaly] DETECTED tables={state.tables}  score={anm_score}")
        rec = new_record(
            tables        = state.tables,
            original_sql  = state.original_sql,
            chosen_order  = state.tables,
            is_anomalous  = True,
            anomaly_score = anm_score,
        )
        return {
            "status":          "anomaly_detected",
            "optimized_query": state.original_sql,
            "choosen_order":   state.tables,
            "xai_explanation": {},
            "anomaly_score":   anm_score,
            "is_anomalous":    True,
            "query_id":        rec.query_id,
            "log_prob_old":    None,    # no PPO action taken for anomalies
        }

    # 3. Layer 1 — PPO forward pass
    _, attention_weights = ai_model(state_tensor)

    xai_explanation = {
        table: round(attention_weights[i].item() * 100, 2)
        for i, table in enumerate(state.tables)
    }
    optimized_order = [t for t, _ in sorted(
        xai_explanation.items(), key=lambda x: x[1], reverse=True)]
    final_sql = f"/*+ Leading({' '.join(optimized_order)}) */ {state.original_sql}"

    # 4. Compute log_prob_old for PPO ratio (Part 7)
    log_prob_old, first_idx = trainer.compute_action_log_prob(
        state_tensor, state.tables, optimized_order)

    # 5. Create analytics record (outcome filled by /feedback)
    rec = new_record(
        tables        = state.tables,
        original_sql  = state.original_sql,
        chosen_order  = optimized_order,
        is_anomalous  = False,
        anomaly_score = anm_score,
    )

    # 6. Store state_tensor + log_prob_old for train_batch() (Part 7)
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
        "anomaly_score":   anm_score,
        "is_anomalous":    False,
        "query_id":        rec.query_id,
        "log_prob_old":    round(log_prob_old, 6),
    }