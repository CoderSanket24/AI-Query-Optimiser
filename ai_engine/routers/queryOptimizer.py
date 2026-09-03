from fastapi import APIRouter
from models.model import QueryState
from agent.ppo_agent import ExplainableJoinOptimizer
from vectorizer.schema_vectorizer import build_state_tensor

router = APIRouter()

# Input dim = 10 (matches the 10-feature vector produced by schema_vectorizer)
# Hidden dim = 32 (network width)
ai_model = ExplainableJoinOptimizer(input_dim=10, hidden_dim=32)

@router.post("/optimize")
async def optimize_query(state: QueryState):
    print(f"Received query with tables: {state.tables}")

    # 1. Build a REAL [N_tables x 10] state tensor from live PostgreSQL schema stats.
    #    Each row encodes: row_count, page_count, col_count, index_count,
    #    has_pk, has_fk, table_size, avg_row_width, row_density, is_in_query
    state_tensor = build_state_tensor(state.tables)

    # 2. Feed the real state into the Explainable AI model
    action_logits, attention_weights = ai_model(state_tensor)

    # 3. Extract the XAI data (Convert tensor weights to standard Python floats)
    xai_explanation = {}
    for i, table in enumerate(state.tables):
        xai_explanation[table] = round(attention_weights[i].item() * 100, 2)

    # 4. Sort tables by the attention weight the model assigned to each —
    #    highest attention = the model thinks this table should be joined first
    sorted_tables = sorted(xai_explanation.items(), key=lambda item: item[1], reverse=True)
    optimized_order = [item[0] for item in sorted_tables]

    # 5. Generate the pg_hint_plan Leading() hint
    hint_string = f"/*+ Leading({ ' '.join(optimized_order) }) */"
    finalSql = f"{hint_string} {state.original_sql}"

    return {
        "status":          "success",
        "optimized_query": finalSql,
        "choosen_order":   optimized_order,
        "xai_explanation": xai_explanation,
    }