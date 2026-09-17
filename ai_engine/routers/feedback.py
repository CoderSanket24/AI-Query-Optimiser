import os
import threading
from collections import deque
from fastapi import APIRouter
from models.feedback_model import FeedbackPayload
from agent.trainer_instance import trainer
from anomaly.detector_instance import detector
from anomaly.isolation_forest import get_query_vector, MIN_SAMPLES_TO_FIT, RETRAIN_EVERY_N
from analytics.analytics_store import complete_record

router = APIRouter()
CONTENTION_PENALTY = 0.1
_replay_buffer: deque = deque(maxlen=1000)
_buffer_lock = threading.Lock()


@router.post("/feedback")
async def receive_feedback(payload: FeedbackPayload):
    contention_factor = 1.0 + (payload.active_connections * CONTENTION_PENALTY)
    reward = -(payload.latency_ms / contention_factor)

    experience = {
        "tables":             payload.tables,
        "chosen_order":       payload.chosen_order,
        "latency_ms":         payload.latency_ms,
        "active_connections": payload.active_connections,
        "reward":             round(reward, 4),
    }

    with _buffer_lock:
        _replay_buffer.append(experience)
        buf_snapshot = list(_replay_buffer)
    n = len(buf_snapshot)

    print(
        f"[Feedback] order={payload.chosen_order} | "
        f"latency={payload.latency_ms:.0f}ms | reward={reward:.2f} | buffer={n}/1000"
    )

    # PPO training step + checkpoint save
    train_metrics = trainer.train_step(experience)
    trainer.save_checkpoint()

    # Complete the analytics record (links outcome to the optimize decision)
    if payload.query_id is not None:
        complete_record(
            query_id          = payload.query_id,
            latency_ms        = payload.latency_ms,
            active_connections= payload.active_connections,
            reward            = round(reward, 4),
            ppo_step          = train_metrics.get("train_step", 0)  if train_metrics else 0,
            ppo_loss          = train_metrics.get("loss", 0.0)       if train_metrics else 0.0,
            ppo_baseline      = train_metrics.get("baseline", 0.0)   if train_metrics else 0.0,
        )

    # Forest refit check: trigger at MIN_SAMPLES, then every RETRAIN_EVERY_N
    forest_retrained = False
    if n >= MIN_SAMPLES_TO_FIT and (n == MIN_SAMPLES_TO_FIT or n % RETRAIN_EVERY_N == 0):
        vectors = _extract_vectors(buf_snapshot)
        if len(vectors) >= MIN_SAMPLES_TO_FIT:
            detector.fit(vectors)
            forest_retrained = True

    return {
        "status":           "received",
        "reward":           round(reward, 4),
        "buffer_size":      n,
        "training":         train_metrics,
        "forest_retrained": forest_retrained,
    }


def _extract_vectors(experiences: list) -> list:
    from vectorizer.schema_vectorizer import build_state_tensor
    vectors = []
    for exp in experiences:
        try:
            vectors.append(get_query_vector(build_state_tensor(exp["tables"])))
        except Exception as e:
            print(f"[Forest] Skipping experience: {e}")
    return vectors


@router.get("/feedback/buffer")
async def inspect_buffer():
    with _buffer_lock:
        experiences = list(_replay_buffer)
    return {"buffer_size": len(experiences), "experiences": experiences}


@router.get("/feedback/stats")
async def training_stats():
    with _buffer_lock:
        buf_size = len(_replay_buffer)
    ckpt = {"exists": trainer.checkpoint_exists, "path": trainer.checkpoint_path}
    if trainer.checkpoint_exists:
        ckpt["size_kb"] = round(os.stat(trainer.checkpoint_path).st_size / 1024, 2)
    return {
        "train_steps":  trainer.train_count,
        "total_loss":   round(trainer.total_loss, 6),
        "baseline":     round(trainer.baseline, 4),
        "buffer_size":  buf_size,
        "checkpoint":   ckpt,
    }


def get_replay_buffer() -> list:
    with _buffer_lock:
        return list(_replay_buffer)