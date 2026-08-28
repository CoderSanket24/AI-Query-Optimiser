from fastapi import APIRouter
from models.model import QueryState

router = APIRouter()

@router.post("/optimize")
async def optimize_query(state: QueryState):
    print(f"{state.tables}")

    optimized_order = list(reversed(state.tables))

    hint_string = f"/*+ Leading({' '.join(optimized_order)}) */"

    finalSql = f"{hint_string} {state.original_sql}"

    return {
        "status":"success",
        "optimized_query":finalSql,
        "choosen_order":optimized_order
    }