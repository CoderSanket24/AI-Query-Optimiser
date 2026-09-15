"""
feedback.py
-----------
POST /feedback       - Receives telemetry, computes reward, trains, saves checkpoint.
GET  /feedback/buffer - Debug: inspect stored experiences.
GET  /feedback/stats  - Debug: training progress + checkpoint info.
"""

import os
import threading
from collections import deque
from fastapi import APIRouter
from models.feedback_model import FeedbackPayload
from agent.trainer_instance import trainer

router = APIRouter()

CONTENTION_PENALTY = 0.1

_replay_buffer: deque = deque(maxlen=1000)
_buffer_lock = threading.Lock()


@router.post("/feedback")
async def receive_feedback(payload: FeedbackPayload):
    """
    1. Compute contention-adjusted reward
    2. Store experience in replay buffer
    3. Trigger one PPO training step
    4. Save checkpoint to disk
    5. Return reward + training metrics
    """
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

    # PPO training step
    train_metrics = trainer.train_step(experience)

    # Persist weights to disk after every update
    trainer.save_checkpoint()

    return {
        "status":      "received",
        "reward":      round(reward, 4),
        "buffer_size": buffer_size,
        "training":    train_metrics,
    }


@router.get("/feedback/buffer")
async def inspect_buffer():
    """Debug: returns all buffered experiences."""
    with _buffer_lock:
        experiences = list(_replay_buffer)
    return {
        "buffer_size": len(experiences),
        "experiences": experiences,
    }


@router.get("/feedback/stats")
async def training_stats():
    """Debug: training progress and checkpoint status."""
    with _buffer_lock:
        buf_size = len(_replay_buffer)

    checkpoint_info = {
        "exists": trainer.checkpoint_exists,
        "path":   trainer.checkpoint_path,
    }
    if trainer.checkpoint_exists:
        stat = os.stat(trainer.checkpoint_path)
        checkpoint_info["size_kb"]       = round(stat.st_size / 1024, 2)
        checkpoint_info["last_modified"] = stat.st_mtime

    return {
        "train_steps":  trainer.train_count,
        "total_loss":   round(trainer.total_loss, 6),
        "baseline":     round(trainer.baseline, 4),
        "buffer_size":  buf_size,
        "checkpoint":   checkpoint_info,
    }


def get_replay_buffer() -> list:
    """Public accessor for future batch training."""
    with _buffer_lock:
        return list(_replay_buffer)