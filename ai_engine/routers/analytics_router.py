"""
analytics_router.py
-------------------
GET /analytics/summary     - aggregate stats (totals, rates, trends)
GET /analytics/history     - paginated query log (newest first)
GET /analytics/improvement - reward trend over last N queries
"""

from fastapi import APIRouter, Query as QParam
from analytics.analytics_store import get_history, get_all, total_count

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/summary")
async def summary():
    records = get_all()
    n = len(records)
    if n == 0:
        return {"total_queries": 0, "message": "No queries recorded yet."}

    anomalous   = [r for r in records if r.is_anomalous]
    normal      = [r for r in records if not r.is_anomalous]

    rewards     = [r.reward     for r in records if r.reward != 0.0]
    latencies   = [r.latency_ms for r in records if r.latency_ms > 0]

    # Reward trend: split into first-half vs second-half
    half = max(1, len(rewards) // 2)
    first_half_avg  = round(sum(rewards[:half])  / half, 4) if rewards else 0
    second_half_avg = round(sum(rewards[half:])  / max(1, len(rewards) - half), 4) if rewards else 0
    learning_delta  = round(second_half_avg - first_half_avg, 4)

    # Table frequency (most optimized)
    from collections import Counter
    table_counts = Counter()
    for r in records:
        for t in r.tables:
            table_counts[t] += 1

    # Best / worst by reward
    completed = [r for r in records if r.reward != 0.0]
    best  = max(completed, key=lambda r: r.reward).to_dict()  if completed else None
    worst = min(completed, key=lambda r: r.reward).to_dict()  if completed else None

    return {
        "total_queries":     n,
        "normal_queries":    len(normal),
        "anomalous_queries": len(anomalous),
        "anomaly_rate_pct":  round(len(anomalous) / n * 100, 1),

        "latency": {
            "avg_ms":  round(sum(latencies) / len(latencies), 1) if latencies else 0,
            "min_ms":  round(min(latencies), 1) if latencies else 0,
            "max_ms":  round(max(latencies), 1) if latencies else 0,
        },

        "reward": {
            "avg":          round(sum(rewards) / len(rewards), 4) if rewards else 0,
            "first_half_avg":  first_half_avg,
            "second_half_avg": second_half_avg,
            "learning_delta":  learning_delta,
            "improving":       learning_delta > 0,
        },

        "top_tables": dict(table_counts.most_common(5)),
        "best_query":  best,
        "worst_query": worst,
    }


@router.get("/history")
async def history(
    limit:  int = QParam(default=20, ge=1, le=200),
    offset: int = QParam(default=0,  ge=0),
):
    items = get_history(limit=limit, offset=offset)
    return {
        "total":   total_count(),
        "limit":   limit,
        "offset":  offset,
        "records": items,
    }


@router.get("/improvement")
async def improvement(n: int = QParam(default=20, ge=2, le=200)):
    """
    Returns the reward for the last N completed queries in chronological order.
    A rising trend means the agent is learning to make better join decisions.
    """
    records = get_all()
    completed = [r for r in records if r.reward != 0.0]
    recent = completed[-n:]

    points = [
        {
            "query_id":    r.query_id,
            "timestamp":   r.timestamp,
            "reward":      r.reward,
            "latency_ms":  r.latency_ms,
            "is_anomalous":r.is_anomalous,
            "ppo_step":    r.ppo_step,
        }
        for r in recent
    ]

    rewards = [p["reward"] for p in points]
    trend   = "improving" if len(rewards) >= 2 and rewards[-1] > rewards[0] else "declining"

    return {
        "n_points":   len(points),
        "trend":      trend,
        "first_reward": round(rewards[0],  4) if rewards else None,
        "last_reward":  round(rewards[-1], 4) if rewards else None,
        "delta":        round(rewards[-1] - rewards[0], 4) if len(rewards) >= 2 else 0,
        "points":     points,
    }
