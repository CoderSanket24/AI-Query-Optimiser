"""
feedback.py
-----------
POST /feedback endpoint.

Java calls this after every executeAndTrack() with real execution telemetry.
This module:
  1. Computes the PPO reward from latency + contention data
  2. Stores the experience in a thread-safe in-memory replay buffer
  3. Exposes get_replay_buffer() for Part 3 (PPO training loop) to consume

Reward formula:
  adjusted_latency = latency_ms / (1 + active_connections * CONTENTION_PENALTY)
  reward = -adjusted_latency

The contention penalty prevents reward poisoning: if the server is busy,
we reduce the negative reward slightly since slow execution is partially
the network load, not the AI choice.
"""

import threading
from collections import deque
from fastapi import APIRouter
from models.feedback_model import FeedbackPayload

router = APIRouter()

# Contention penalty weight — tune between 0.0 (ignore load) and 1.0 (full penalty)
CONTENTION_PENALTY = 0.1

# Thread-safe replay buffer — stores the last 1000 experiences
# (Part 3 PPO training loop will drain this)
_replay_buffer: deque = deque(maxlen=1000)
_buffer_lock = threading.Lock()


@router.post("/feedback")
async def receive_feedback(payload: FeedbackPayload):
    """
    Receives execution telemetry from Java middleware after each query.
    Computes a reward and stores the experience in the replay buffer.
    """
    # --- Reward computation -------------------------------------------
    # Reduce the latency penalty when server is under contention,
    # so the agent is not penalised for congestion it did not cause.
    contention_factor = 1.0 + (payload.active_connections * CONTENTION_PENALTY)
    adjusted_latency  = payload.latency_ms / contention_factor
    reward = -adjusted_latency   # more negative = worse plan

    experience = {
        "tables":            payload.tables,
        "chosen_order":      payload.chosen_order,
        "latency_ms":        payload.latency_ms,
        "active_connections": payload.active_connections,
        "reward":            round(reward, 4),
    }

    with _buffer_lock:
        _replay_buffer.append(experience)
        buffer_size = len(_replay_buffer)

    print(
        f"[Feedback] Received | "
        f"order={payload.chosen_order} | "
        f"latency={payload.latency_ms:.0f}ms | "
        f"connections={payload.active_connections} | "
        f"reward={reward:.2f} | "
        f"buffer={buffer_size}/1000"
    )

    return {
        "status":      "received",
        "reward":      round(reward, 4),
        "buffer_size": buffer_size,
    }


@router.get("/feedback/buffer")
async def inspect_buffer():
    """
    Debug endpoint: returns the current replay buffer contents.
    Useful for verifying the feedback loop is working before Part 3.
    """
    with _buffer_lock:
        experiences = list(_replay_buffer)

    return {
        "buffer_size": len(experiences),
        "experiences": experiences,
    }


def get_replay_buffer() -> list:
    """
    Public accessor for Part 3 (PPO training loop).
    Returns a snapshot of all buffered experiences.
    """
    with _buffer_lock:
        return list(_replay_buffer)