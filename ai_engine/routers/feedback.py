"""
feedback.py
-----------
POST /feedback  - Receives execution telemetry from Java, computes the PPO
                  reward, stores the experience, and triggers one training step.
GET  /feedback/buffer - Debug: inspect stored experiences.
GET  /feedback/stats  - Debug: inspect PPO training progress.
"""

import threading
from collections import deque
from fastapi import APIRouter
from models.feedback_model import FeedbackPayload
from agent.trainer_instance import trainer

router = APIRouter()

# Contention penalty: reduce the negative reward when other connections are
# active (slow execution may be server load, not a bad join plan).
CONTENTION_PENALTY = 0.1

# Thread-safe replay buffer (max 1000 experiences)
_replay_buffer: deque = deque(maxlen=1000)
_buffer_lock = threading.Lock()


@router.post("/feedback")
async def receive_feedback(payload: FeedbackPayload):
    """
    1. Compute contention-adjusted reward
    2. Store experience in replay buffer
    3. Trigger one PPO training step immediately
    4. Return reward + training metrics to Java for logging
    """
    # --- Reward -----------------------------------------------------------
    contention_factor = 1.0 + (payload.active_connections * CONTENTION_PENALTY)
    adjusted_latency  = payload.latency_ms / contention_factor
    reward = -adjusted_latency

    experience = {
        "tables":             payload.tables,
        "chosen_order":       payload.chosen_order,
        "latency_ms":         payload.latency_ms,
        "active_connections": payload.active_connections,
        "reward":             round(reward, 4),
    }

    with _buffer_lock:
        _replay_buffer.append(experience)
        buffer_size = len(_replay_buffer)

    print(
        f"[Feedback] order={payload.chosen_order} | "
        f"latency={payload.latency_ms:.0f}ms | "
        f"connections={payload.active_connections} | "
        f"reward={reward:.2f} | buffer={buffer_size}/1000"
    )

    # --- PPO Training Step -----------------------------------------------
    train_metrics = trainer.train_step(experience)

    return {
        "status":        "received",
        "reward":        round(reward, 4),
        "buffer_size":   buffer_size,
        "training":      train_metrics,
    }


@router.get("/feedback/buffer")
async def inspect_buffer():
    """Debug endpoint: returns all buffered experiences."""
    with _buffer_lock:
        experiences = list(_replay_buffer)
    return {
        "buffer_size": len(experiences),
        "experiences": experiences,
    }


@router.get("/feedback/stats")
async def training_stats():
    """Debug endpoint: returns PPO training progress metrics."""
    with _buffer_lock:
        buf_size = len(_replay_buffer)
    return {
        "train_steps":  trainer.train_count,
        "total_loss":   round(trainer.total_loss, 6),
        "baseline":     round(trainer.baseline, 4),
        "buffer_size":  buf_size,
    }


def get_replay_buffer() -> list:
    """Public accessor for future batch training."""
    with _buffer_lock:
        return list(_replay_buffer)