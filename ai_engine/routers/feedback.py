"""
feedback.py  (Part 7 — True PPO Batch Training)
-------------------------------------------------
POST /feedback
  1. Compute reward from latency + contention
  2. Complete analytics record
  3. If log_prob_old present: accumulate into PPO batch buffer
       → when batch reaches BATCH_SIZE: call trainer.train_batch() (PPO clip)
     Else (anomaly path): call trainer.train_step() (REINFORCE fallback)
  4. Forest refit check
  5. Checkpoint save

GET /feedback/stats
GET /feedback/buffer
"""

import os
import threading
from collections import deque
from fastapi import APIRouter
from models.feedback_model import FeedbackPayload
from agent.trainer_instance import trainer
from agent.experience_buffer import pop_pending
from anomaly.detector_instance import detector
from anomaly.isolation_forest import get_query_vector, MIN_SAMPLES_TO_FIT, RETRAIN_EVERY_N
from analytics.analytics_store import complete_record

router = APIRouter()

CONTENTION_PENALTY = 0.1
BATCH_SIZE         = 8       # collect 8 experiences before each PPO batch update

# Replay buffer (for forest refit + /feedback/buffer debug endpoint)
_replay_buffer: deque = deque(maxlen=1000)
_buffer_lock = threading.Lock()

# PPO batch accumulator (Part 7)
_ppo_batch:      list = []
_ppo_batch_lock  = threading.Lock()


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

    # ── Training ──────────────────────────────────────────────────────
    train_metrics = {}
    ppo_batch_fired = False

    if payload.log_prob_old is not None and payload.query_id is not None:
        # Part 7: True PPO batch path — retrieve stored state_tensor
        pending = pop_pending(payload.query_id)
        if pending is not None:
            ppo_exp = {
                "state_tensor": pending["state_tensor"],
                "first_idx":    pending["first_idx"],
                "log_prob_old": payload.log_prob_old,
                "reward":       round(reward, 4),
            }
            with _ppo_batch_lock:
                _ppo_batch.append(ppo_exp)
                batch_ready = len(_ppo_batch) >= BATCH_SIZE
                if batch_ready:
                    batch_to_train = _ppo_batch[:]
                    _ppo_batch.clear()

            if batch_ready:
                train_metrics   = trainer.train_batch(batch_to_train)
                ppo_batch_fired = True
                trainer.save_checkpoint()
                print(f"[PPO/Clip] Batch update fired  size={len(batch_to_train)}")
            else:
                with _ppo_batch_lock:
                    queued = len(_ppo_batch)
                train_metrics = {"mode": "ppo_accumulating", "queued": queued, "batch_size": BATCH_SIZE}
                print(f"[PPO/Clip] Accumulating batch {queued}/{BATCH_SIZE}")
        else:
            # Pending expired — fall back to REINFORCE
            train_metrics = trainer.train_step(experience)
            trainer.save_checkpoint()
    else:
        # Anomaly path or old Java client — use REINFORCE
        train_metrics = trainer.train_step(experience)
        trainer.save_checkpoint()

    # ── Analytics record completion ───────────────────────────────────
    if payload.query_id is not None:
        complete_record(
            query_id           = payload.query_id,
            latency_ms         = payload.latency_ms,
            active_connections = payload.active_connections,
            reward             = round(reward, 4),
            ppo_step           = train_metrics.get("train_step", trainer.train_count),
            ppo_loss           = train_metrics.get("loss",       train_metrics.get("avg_loss", 0.0)),
            ppo_baseline       = train_metrics.get("baseline",   trainer.baseline),
        )

    # ── Forest refit ─────────────────────────────────────────────────
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
        "ppo_batch_fired":  ppo_batch_fired,
        "forest_retrained": forest_retrained,
    }


def _extract_vectors(experiences: list) -> list:
    from vectorizer.schema_vectorizer import build_state_tensor
    vectors = []
    for exp in experiences:
        try:
            vectors.append(get_query_vector(build_state_tensor(exp["tables"])))
        except Exception as e:
            print(f"[Forest] Skipping: {e}")
    return vectors


@router.get("/feedback/buffer")
async def inspect_buffer():
    with _buffer_lock:
        experiences = list(_replay_buffer)
    with _ppo_batch_lock:
        ppo_queued = len(_ppo_batch)
    return {
        "buffer_size": len(experiences),
        "ppo_batch_queued": ppo_queued,
        "ppo_batch_size_target": BATCH_SIZE,
        "experiences": experiences,
    }


@router.get("/feedback/stats")
async def training_stats():
    with _buffer_lock:
        buf_size = len(_replay_buffer)
    with _ppo_batch_lock:
        ppo_queued = len(_ppo_batch)
    ckpt = {"exists": trainer.checkpoint_exists, "path": trainer.checkpoint_path}
    if trainer.checkpoint_exists:
        ckpt["size_kb"] = round(os.stat(trainer.checkpoint_path).st_size / 1024, 2)
    return {
        "train_steps":        trainer.train_count,
        "total_loss":         round(trainer.total_loss, 6),
        "baseline":           round(trainer.baseline, 4),
        "buffer_size":        buf_size,
        "ppo_batch_queued":   ppo_queued,
        "ppo_batch_size":     BATCH_SIZE,
        "checkpoint":         ckpt,
    }


def get_replay_buffer() -> list:
    with _buffer_lock:
        return list(_replay_buffer)