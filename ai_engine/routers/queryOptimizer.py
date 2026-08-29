from fastapi import APIRouter
from models.model import QueryState
import torch
from agent.ppo_agent import ExplainableJoinOptimizer

router = APIRouter()

# Initialize our AI Model (Using dummy dimensions for now until we build the vectorizer)
# We assume a max of 10 tables, with a hidden network size of 32
ai_model = ExplainableJoinOptimizer(input_dim=10, hidden_dim=32)

@router.post("/optimize")
async def optimize_query(state: QueryState):
    print(f"Received query with tables: {state.tables}")

    # 1. Convert the list of tables into a dummy mathematical tensor for the AI
    # (In Milestone 5, we will map real table schemas to these numbers)
    state_tensor = torch.rand((len(state.tables), 10)) 
    
    # 2. Feed the state into the Explainable AI model
    action_logits, attention_weights = ai_model(state_tensor)
    
    # 3. Extract the XAI data (Convert tensor weights to standard Python floats)
    xai_explanation = {}
    for i, table in enumerate(state.tables):
        xai_explanation[table] = round(attention_weights[i].item() * 100, 2)
    
    # 4. For now, we simulate the AI's chosen order based on its highest attention focus
    # We sort the tables based on which one the AI paid the most attention to
    sorted_tables = sorted(xai_explanation.items(), key=lambda item: item[1], reverse=True)
    optimized_order = [item[0] for item in sorted_tables]
    
    # 5. Generate the pg_hint_plan syntax
    hint_string = f"/*+ Leading({ ' '.join(optimized_order) }) */"
    finalSql = f"{hint_string} {state.original_sql}"

    return {
        "status":"success",
        "optimized_query":finalSql,
        "choosen_order":optimized_order,
        "xai_explanation":xai_explanation
    }