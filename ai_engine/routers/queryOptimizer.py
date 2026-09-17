from fastapi import APIRouter
from models.model import QueryState
from agent.trainer_instance import trainer
from vectorizer.schema_vectorizer import build_state_tensor
from anomaly.detector_instance import detector
from anomaly.isolation_forest import get_query_vector

router   = APIRouter()
ai_model = trainer.model


@router.post("/optimize")
async def optimize_query(state: QueryState):
    print(f"[Optimize] Tables: {state.tables}")

    # 1. Build real state tensor from live schema stats
    state_tensor = build_state_tensor(state.tables)

    # 2. Layer 2 — Anomaly detection (no-op until forest fitted with MIN_SAMPLES)
    query_vector          = get_query_vector(state_tensor)
    is_anomaly, anm_score = detector.predict(query_vector)

    if is_anomaly:
        print(f"[Anomaly] DETECTED tables={state.tables}  score={anm_score}")
        return {
            "status":          "anomaly_detected",
            "optimized_query": state.original_sql,   # original SQL — no hint
            "choosen_order":   state.tables,          # unchanged
            "xai_explanation": {},
            "anomaly_score":   anm_score,
            "is_anomalous":    True,
        }

    # 3. Layer 1 — PPO forward pass
    _, attention_weights = ai_model(state_tensor)

    xai_explanation = {
        table: round(attention_weights[i].item() * 100, 2)
        for i, table in enumerate(state.tables)
    }
    optimized_order = [t for t, _ in sorted(xai_explanation.items(), key=lambda x: x[1], reverse=True)]
    final_sql       = f"/*+ Leading({' '.join(optimized_order)}) */ {state.original_sql}"

    return {
        "status":          "success",
        "optimized_query": final_sql,
        "choosen_order":   optimized_order,
        "xai_explanation": xai_explanation,
        "anomaly_score":   anm_score,   # near 0.0 means normal
        "is_anomalous":    False,
    }