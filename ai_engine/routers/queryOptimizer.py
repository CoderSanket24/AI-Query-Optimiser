from fastapi import APIRouter
from models.model import QueryState
from agent.trainer_instance import trainer
from vectorizer.schema_vectorizer import build_state_tensor

router = APIRouter()

# trainer.model is the shared ExplainableJoinOptimizer instance.
# PPO training in /feedback updates its weights; /optimize reads them.
ai_model = trainer.model


@router.post("/optimize")
async def optimize_query(state: QueryState):
    print(f"Received query with tables: {state.tables}")

    # 1. Build real [N_tables x 10] state tensor from live PostgreSQL schema stats
    state_tensor = build_state_tensor(state.tables)

    # 2. Forward pass through the (continuously trained) model
    action_logits, attention_weights = ai_model(state_tensor)

    # 3. Extract XAI attention weights as percentages
    xai_explanation = {}
    for i, table in enumerate(state.tables):
        xai_explanation[table] = round(attention_weights[i].item() * 100, 2)

    # 4. Sort by attention weight: highest focus = join first
    sorted_tables  = sorted(xai_explanation.items(), key=lambda x: x[1], reverse=True)
    optimized_order = [item[0] for item in sorted_tables]

    # 5. Generate pg_hint_plan Leading() hint
    hint_string = f"/*+ Leading({ ' '.join(optimized_order) }) */"
    final_sql   = f"{hint_string} {state.original_sql}"

    return {
        "status":          "success",
        "optimized_query": final_sql,
        "choosen_order":   optimized_order,
        "xai_explanation": xai_explanation,
    }