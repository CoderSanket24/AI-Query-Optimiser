"""
feedback.py  (Part 5 v2 + Part 7)
-----------------------------------
POST /feedback
  Receives per-query telemetry after execution.

  Step 1 — Isolation Forest anomaly check (DoS detection):
    Features: [wait_time_ms, exec_time_ms, active_connections]
    Normal:   high wait + high connections  (legitimate DB load)
    Anomaly:  high exec  + low connections  (hacker's slow query)
              high wait  + low connections  (victims behind DoS)

    If ANOMALY:
      reward = 0.0               ← neutralise, PPO learns NOTHING
      PPO training skipped       ← protects RL from poisoned signal
      analytics record updated   ← marked is_anomalous=True
      log warning

    If NORMAL:
      reward = -exec_time_ms / (1 + connections × 0.1)
               ← exec_time only (join order does NOT affect wait_time)
      PPO batch accumulation     ← train when batch_size=8 reached

  Step 2 — Forest refit (every MIN_SAMPLES / RETRAIN_EVERY_N samples).
  Step 3 — Checkpoint save (after every PPO batch update).

GET /feedback/stats   -- PPO + buffer + checkpoint state
GET /feedback/buffer  -- raw replay buffer (debug)
"""

import os
import threading
from collections import deque
from fastapi import APIRouter
from models.feedback_model import FeedbackPayload
from agent.trainer_instance import trainer
from agent.experience_buffer import pop_pending
from anomaly.detector_instance import detector
from anomaly.isolation_forest import MIN_SAMPLES_TO_FIT, RETRAIN_EVERY_N
from analytics.analytics_store import complete_record

router = APIRouter()

CONTENTION_PENALTY = 0.1
BATCH_SIZE         = 8

_replay_buffer: deque = deque(maxlen=1000)
_buffer_lock          = threading.Lock()

_ppo_batch:     list  = []
_ppo_batch_lock       = threading.Lock()

# Accumulate telemetry samples for forest refit
_forest_samples: list = []
_forest_lock          = threading.Lock()


@router.post("/feedback")
async def receive_feedback(payload: FeedbackPayload):
    exec_ms = payload.exec_time_ms if payload.exec_time_ms > 0 else payload.latency_ms
    wait_ms = payload.wait_time_ms

    # ── Step 1: Isolation Forest anomaly check ──────────────────────
    is_anomaly, anomaly_score = detector.predict(
        wait_time_ms       = wait_ms,
        exec_time_ms       = exec_ms,
        active_connections = payload.active_connections,
    )

    if is_anomaly:
        # DoS / bad query detected — zero reward, skip PPO
        reward = 0.0
        print(
            f"[ANOMALY] DoS detected! "
            f"wait={wait_ms:.0f}ms  exec={exec_ms:.0f}ms  "
            f"conn={payload.active_connections}  score={anomaly_score}"
        )
        train_metrics    = {
            "mode":          "skipped_anomaly",
            "reason":        "DoS pattern detected by Isolation Forest",
            "anomaly_score": anomaly_score,
        }
        ppo_batch_fired  = False

        # Update analytics with anomaly flag
        if payload.query_id is not None:
            complete_record(
                query_id           = payload.query_id,
                latency_ms         = payload.latency_ms,
                active_connections = payload.active_connections,
                reward             = reward,
                ppo_step           = trainer.train_count,
                ppo_loss           = 0.0,
                ppo_baseline       = trainer.baseline,
            )

        # Still accumulate this sample for forest self-improvement
        _accumulate_forest_sample(wait_ms, exec_ms, payload.active_connections)

        return {
            "status":           "received",
            "anomaly_detected": True,
            "anomaly_score":    anomaly_score,
            "reward":           reward,
            "buffer_size":      len(_replay_buffer),
            "training":         train_metrics,
            "ppo_batch_fired":  False,
            "forest_retrained": False,
        }

    # ── Step 2: Normal query — compute reward using exec_time only ───
    # Join order affects exec_time, NOT wait_time → use exec_time for reward
    contention_factor = 1.0 + (payload.active_connections * CONTENTION_PENALTY)
    reward = -(exec_ms / contention_factor)

    experience = {
        "tables":             payload.tables,
        "chosen_order":       payload.chosen_order,
        "latency_ms":         payload.latency_ms,
        "exec_time_ms":       exec_ms,
        "wait_time_ms":       wait_ms,
        "active_connections": payload.active_connections,
        "reward":             round(reward, 4),
    }

    with _buffer_lock:
        _replay_buffer.append(experience)
        buf_snapshot = list(_replay_buffer)
    n = len(buf_snapshot)

    print(
        f"[Feedback] order={payload.chosen_order} | "
        f"wait={wait_ms:.0f}ms exec={exec_ms:.0f}ms | "
        f"reward={reward:.2f} | buffer={n}/1000"
    )

    # ── Step 3: PPO training ─────────────────────────────────────────
    train_metrics   = {}
    ppo_batch_fired = False

    if payload.log_prob_old is not None and payload.query_id is not None:
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
            else:
                with _ppo_batch_lock:
                    queued = len(_ppo_batch)
                train_metrics = {
                    "mode":       "ppo_accumulating",
                    "queued":     queued,
                    "batch_size": BATCH_SIZE,
                }
        else:
            train_metrics = trainer.train_step(experience)
            trainer.save_checkpoint()
    else:
        train_metrics = trainer.train_step(experience)
        trainer.save_checkpoint()

    # ── Step 4: Analytics ────────────────────────────────────────────
    if payload.query_id is not None:
        complete_record(
            query_id           = payload.query_id,
            latency_ms         = payload.latency_ms,
            active_connections = payload.active_connections,
            reward             = round(reward, 4),
            ppo_step           = train_metrics.get("train_step", trainer.train_count),
            ppo_loss           = train_metrics.get("loss", train_metrics.get("avg_loss", 0.0)),
            ppo_baseline       = train_metrics.get("baseline", trainer.baseline),
        )

    # ── Step 5: Forest refit ─────────────────────────────────────────
    forest_retrained = _accumulate_forest_sample(
        wait_ms, exec_ms, payload.active_connections)

    return {
        "status":           "received",
        "anomaly_detected": False,
        "anomaly_score":    anomaly_score,
        "reward":           round(reward, 4),
        "exec_time_ms":     exec_ms,
        "wait_time_ms":     wait_ms,
        "buffer_size":      n,
        "training":         train_metrics,
        "ppo_batch_fired":  ppo_batch_fired,
        "forest_retrained": forest_retrained,
    }


def _accumulate_forest_sample(wait_ms: float,
                               exec_ms: float,
                               connections: int) -> bool:
    """
    Add one [wait_ms, exec_ms, connections] sample to the forest buffer.
    Triggers refit when MIN_SAMPLES_TO_FIT reached, then every RETRAIN_EVERY_N.
    Returns True if forest was retrained.
    """
    sample = [wait_ms, exec_ms, float(connections)]
    with _forest_lock:
        _forest_samples.append(sample)
        n       = len(_forest_samples)
        samples = list(_forest_samples)

    should_fit = (n == MIN_SAMPLES_TO_FIT or
                  (n > MIN_SAMPLES_TO_FIT and (n - MIN_SAMPLES_TO_FIT) % RETRAIN_EVERY_N == 0))

    if should_fit and n >= MIN_SAMPLES_TO_FIT:
        detector.fit(samples)
        return True
    return False


# ── Debug endpoints ──────────────────────────────────────────────────

@router.get("/feedback/buffer")
async def inspect_buffer():
    with _buffer_lock:
        experiences = list(_replay_buffer)
    with _ppo_batch_lock:
        ppo_queued = len(_ppo_batch)
    with _forest_lock:
        forest_samples = len(_forest_samples)
    return {
        "buffer_size":           len(experiences),
        "ppo_batch_queued":      ppo_queued,
        "ppo_batch_size_target": BATCH_SIZE,
        "forest_samples":        forest_samples,
        "forest_min_to_fit":     MIN_SAMPLES_TO_FIT,
        "experiences":           experiences,
    }


@router.get("/feedback/stats")
async def training_stats():
    with _buffer_lock:
        buf_size = len(_replay_buffer)
    with _ppo_batch_lock:
        ppo_queued = len(_ppo_batch)
    with _forest_lock:
        forest_samples = len(_forest_samples)
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
        "forest_samples":     forest_samples,
        "forest_min_to_fit":  MIN_SAMPLES_TO_FIT,
        "checkpoint":         ckpt,
    }


def get_replay_buffer() -> list:
    with _buffer_lock:
        return list(_replay_buffer)