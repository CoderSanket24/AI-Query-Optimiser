"""
feedback.py  (Part 5 v2 + Part 7)
-----------------------------------
POST /feedback
  Receives per-query telemetry after execution.

  Step 1 — Isolation Forest anomaly check (DoS detection):
    Features: [wait_time_ms, active_connections]   (exec_time EXCLUDED)
    Normal:   low wait + any connections    (queries going through fine)
              high wait + high connections  (legitimate DB load)
    Anomaly:  high wait + LOW connections   (victims waiting behind DoS)
    Note: exec_time excluded -- it can be high due to PPO's own bad join
          order, which would falsely block PPO from learning.

    If ANOMALY:
      reward = 0.0               ← neutralise, PPO learns NOTHING
      PPO training skipped       ← protects RL from poisoned signal
      analytics record updated   ← marked is_anomalous=True
      flagged sample QUARANTINED ← NOT fed back to forest to prevent
                                   the detector from learning attacks are normal
      log warning

    If NORMAL:
      reward = -exec_time_ms / (1 + connections × 0.1)
               ← exec_time only (join order does NOT affect wait_time)
               ← exec_time_ms == EXEC_MISSING (-1) → REINFORCE fallback,
                  reward not computed (fast-query timing gap)
      PPO batch accumulation     ← train when batch_size=8 reached

  Step 2 — Forest refit (every MIN_SAMPLES / RETRAIN_EVERY_N NORMAL samples).
  Step 3 — Checkpoint save (after every PPO batch update).

  OFFLOAD (Part 2b-1):
    Training (train_batch/train_step), checkpoint save, and forest refit are
    moved OFF the HTTP request path onto a single daemon worker thread that
    drains a bounded queue (QUEUE_MAX).  The request thread only computes the
    reward, accumulates the batch, and enqueues the work; a full queue DROPS
    the item (counted, never blocks).  Because the worker is a daemon, any
    items still queued at process exit are LOST (that train/refit/checkpoint
    never runs).

  PPO path (Bug 3 fix):
    log_prob_old is read from the server-side pending store (pop_pending),
    NOT from the HTTP payload.  The client never sees or sends log_prob_old,
    so it cannot be tampered with.  The payload field `log_prob_old` has been
    removed; the condition now only checks query_id.

Per-stage latency instrumentation
----------------------------------
server_timings_ms keys (all fractional ms via time.perf_counter()):
  anomaly_predict  -- detector.predict() call
  training         -- always 0.0 now (training is offloaded to the worker)
  checkpoint_save  -- always 0.0 now (checkpoint save is offloaded to the worker)
  analytics        -- complete_record() call
  forest_refit     -- always 0.0 now (refit is offloaded to the worker)
  feedback_total   -- entire endpoint wall-clock (superset of the above)

  The worker logs its own wall-clock as fb_worker_train_ms (train + checkpoint).
  Because analytics is completed at request time, analytics records show the
  PRE-TRAINING train_count / loss / baseline (they lag one training event).

GET /feedback/stats   -- PPO + buffer + checkpoint + worker queue/drop/error state
GET /feedback/buffer  -- raw replay buffer (debug)
"""

import os
import time
import queue
import threading
import traceback
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

# Sentinel value: exec_time could not be measured (nanoTime path was skipped).
# Distinct from 0 ms which is a valid measurement for extremely fast queries.
EXEC_MISSING = -1.0

_replay_buffer: deque = deque(maxlen=1000)
_buffer_lock          = threading.Lock()

_ppo_batch:     list  = []
_ppo_batch_lock       = threading.Lock()

# Normal-query-only samples for forest refit.
# Anomalous samples are QUARANTINED — never added here.
_forest_samples: list = []
_forest_lock          = threading.Lock()


# ── Background training / checkpoint / forest-refit worker ──────────
# All model and forest WRITES are moved off the request path onto a single
# daemon worker thread that drains a bounded queue.  When the queue is full
# the request DROPS the item (and counts it) instead of ever blocking.
#
# NOTE: a daemon thread does not drain on shutdown, so any items still queued
# when the process exits are LOST (that training/refit/checkpoint never runs).
QUEUE_MAX = 256

_worker_queue         = queue.Queue(maxsize=QUEUE_MAX)
_stats_lock           = threading.Lock()
_dropped_count        = 0
_worker_error_count   = 0
_last_worker_error    = ""
_last_worker_train_ms = 0.0


def _process_item(kind: str, payload) -> float:
    """Run one queued work item. Returns elapsed ms. Raises on error."""
    _t0 = time.perf_counter()
    if kind == "ppo_train":
        trainer.train_batch(payload)
        trainer.save_checkpoint()
    elif kind == "reinforce":
        trainer.train_step(payload)
        trainer.save_checkpoint()
    elif kind == "forest_refit":
        detector.fit(payload)
    elif kind == "checkpoint":
        trainer.save_checkpoint()
    else:
        raise ValueError(f"unknown worker item kind: {kind}")
    return (time.perf_counter() - _t0) * 1000.0


def _worker_loop(q: queue.Queue) -> None:
    """Drain `q` forever. Exceptions are logged and never kill the worker."""
    global _worker_error_count, _last_worker_error, _last_worker_train_ms
    while True:
        item = q.get()
        kind = item[0]
        try:
            elapsed_ms = _process_item(kind, item[1])
            if kind in ("ppo_train", "reinforce", "checkpoint"):
                with _stats_lock:
                    _last_worker_train_ms = elapsed_ms
                print(f"[Worker] {kind} done  fb_worker_train_ms={elapsed_ms:.3f}")
            else:
                print(f"[Worker] {kind} done in {elapsed_ms:.3f} ms")
        except Exception as exc:
            with _stats_lock:
                _worker_error_count += 1
                _last_worker_error = f"{type(exc).__name__}: {exc}"
            print(f"[Worker] ERROR in {kind}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        finally:
            q.task_done()


# The queue object is captured by the thread's target so that swapping the
# module-global `_worker_queue` (done by tests) never disrupts the worker.
_worker_thread = threading.Thread(
    target=_worker_loop, args=(_worker_queue,),
    name="feedback-worker", daemon=True)
_worker_thread.start()


def _enqueue(item) -> bool:
    """Non-blocking enqueue. Returns True if queued, False if dropped (full)."""
    global _dropped_count
    try:
        _worker_queue.put_nowait(item)
        return True
    except queue.Full:
        with _stats_lock:
            _dropped_count += 1
        print("[Worker] queue full — dropped one item (counted)")
        return False


@router.post("/feedback")
async def receive_feedback(payload: FeedbackPayload):
    _fb_start = time.perf_counter()

    # ── exec_ms resolution ───────────────────────────────────────────
    # exec_time_ms == EXEC_MISSING (-1) means the Java side could not measure it
    # (should be rare with nanoTime, but guard anyway).
    # We do NOT fall back to latency_ms because that includes wait_time,
    # which the join order cannot affect.
    exec_ms = payload.exec_time_ms   # may be EXEC_MISSING
    wait_ms = payload.wait_time_ms

    exec_is_missing = (exec_ms == EXEC_MISSING)

    # ── Step 1: Isolation Forest anomaly check ──────────────────────
    _t0 = time.perf_counter()
    is_anomaly, anomaly_score = detector.predict(
        wait_time_ms       = wait_ms,
        active_connections = payload.active_connections,
    )
    anomaly_predict_ms = (time.perf_counter() - _t0) * 1000.0

    # Timer accumulators. training / checkpoint_save / forest_refit stay 0.0:
    # those stages are offloaded to the worker (logged there as fb_worker_train_ms).
    training_ms       = 0.0
    checkpoint_save_ms = 0.0
    analytics_ms      = 0.0
    forest_refit_ms   = 0.0

    if is_anomaly:
        # DoS / bad query detected — zero reward, skip PPO.
        reward = 0.0
        print(
            f"[ANOMALY] DoS detected! "
            f"wait={wait_ms:.0f}ms  exec={'?' if exec_is_missing else f'{exec_ms:.0f}ms'}  "
            f"conn={payload.active_connections}  score={anomaly_score}"
        )
        train_metrics   = {
            "mode":          "skipped_anomaly",
            "reason":        "DoS pattern detected by Isolation Forest",
            "anomaly_score": anomaly_score,
        }
        ppo_batch_fired = False

        # Update analytics — mark as anomalous so history table is accurate
        if payload.query_id is not None:
            _t0 = time.perf_counter()
            complete_record(
                query_id           = payload.query_id,
                latency_ms         = payload.latency_ms,
                active_connections = payload.active_connections,
                reward             = reward,
                ppo_step           = trainer.train_count,
                ppo_loss           = 0.0,
                ppo_baseline       = trainer.baseline,
                is_anomalous       = True,          # ← DoS detected
                anomaly_score      = anomaly_score, # ← Forest score
            )
            analytics_ms = (time.perf_counter() - _t0) * 1000.0

        # QUARANTINE: do NOT add flagged samples to _forest_samples.
        # Feeding anomalies back would teach the forest that attacks are normal.

        feedback_total_ms = (time.perf_counter() - _fb_start) * 1000.0
        return {
            "status":           "received",
            "anomaly_detected": True,
            "anomaly_score":    anomaly_score,
            "reward":           reward,
            "buffer_size":      len(_replay_buffer),
            "training":         train_metrics,
            "ppo_batch_fired":  False,
            "forest_retrained": False,
            "server_timings_ms": {
                "anomaly_predict":  round(anomaly_predict_ms,  3),
                "training":         0.0,
                "checkpoint_save":  0.0,
                "analytics":        round(analytics_ms,        3),
                "forest_refit":     0.0,
                "feedback_total":   round(feedback_total_ms,   3),
            },
        }

    # ── Step 2: Normal query — compute reward using exec_time only ───
    # Join order affects exec_time, NOT wait_time.
    if exec_is_missing:
        # Can't compute a valid reward — record the experience but skip PPO.
        reward = None
        print(
            f"[Feedback] exec_time missing (nanoTime sentinel) — "
            f"wait={wait_ms:.0f}ms  conn={payload.active_connections}  "
            f"→ REINFORCE skipped, no reward computed"
        )
    else:
        contention_factor = 1.0 + (payload.active_connections * CONTENTION_PENALTY)
        reward = -(exec_ms / contention_factor)

    experience = {
        "tables":             payload.tables,
        "stat_names":         payload.real_names if payload.real_names else payload.tables,
        "join_conditions":    getattr(payload, "join_conditions", []) or [],
        "chosen_order":       payload.chosen_order,
        "latency_ms":         payload.latency_ms,
        "exec_time_ms":       exec_ms,
        "wait_time_ms":       wait_ms,
        "active_connections": payload.active_connections,
        "reward":             round(reward, 4) if reward is not None else None,
    }

    if reward is not None:
        with _buffer_lock:
            _replay_buffer.append(experience)
            buf_snapshot = list(_replay_buffer)
        n = len(buf_snapshot)

        print(
            f"[Feedback] order={payload.chosen_order} | "
            f"wait={wait_ms:.0f}ms exec={exec_ms:.0f}ms | "
            f"reward={reward:.2f} | buffer={n}/1000"
        )
    else:
        with _buffer_lock:
            n = len(_replay_buffer)

    # ── Step 3: PPO training (OFFLOADED to the worker) ───────────────
    # Training + checkpoint writes no longer run on the request thread: we
    # compute the reward, accumulate the batch, then ENQUEUE the work.
    # The worker thread performs train_batch/train_step + save_checkpoint.
    # training_ms / checkpoint_save_ms therefore stay 0.0 here — the worker
    # logs its own wall-clock as fb_worker_train_ms.
    train_metrics   = {}
    ppo_batch_fired = False

    if reward is None:
        # Missing exec_time → cannot compute a valid reward; skip training.
        train_metrics = {"mode": "skipped_missing_exec", "reason": "exec_time_ms sentinel (-1)"}

    elif payload.query_id is not None:
        # Read log_prob_old from SERVER-SIDE pending store (not from payload).
        # This prevents client tampering and fixes the PPO-never-runs bug:
        # the value was previously commented out of the /optimize response.
        pending = pop_pending(payload.query_id)
        if pending is not None:
            ppo_exp = {
                "state_tensor":    pending["state_tensor"],
                "aliases":         pending["aliases"],
                "chosen_order":    payload.chosen_order,
                "join_conditions": pending.get("join_conditions", []),  # graph edges
                "log_prob_old":    pending["log_prob_old"],   # ← from server store
                "reward":          round(reward, 4),
            }
            with _ppo_batch_lock:
                _ppo_batch.append(ppo_exp)
                batch_ready = len(_ppo_batch) >= BATCH_SIZE
                if batch_ready:
                    batch_to_train = _ppo_batch[:]
                    _ppo_batch.clear()

            if batch_ready:
                queued = _enqueue(("ppo_train", batch_to_train))
                ppo_batch_fired = True
                train_metrics = {
                    "mode":       "queued" if queued else "queued_full_dropped",
                    "batch_size": len(batch_to_train),
                }
            else:
                with _ppo_batch_lock:
                    queued = len(_ppo_batch)
                train_metrics = {
                    "mode":       "ppo_accumulating",
                    "queued":     queued,
                    "batch_size": BATCH_SIZE,
                }
        else:
            # Pending entry expired or was never created → REINFORCE fallback
            queued = _enqueue(("reinforce", experience))
            train_metrics = {"mode": "queued_reinforce" if queued else "queued_full_dropped"}
    else:
        queued = _enqueue(("reinforce", experience))
        train_metrics = {"mode": "queued_reinforce" if queued else "queued_full_dropped"}

    # ── Step 4: Analytics ────────────────────────────────────────────
    # Completed at REQUEST time, so ppo_step/ppo_loss/ppo_baseline reflect
    # the trainer state BEFORE this request's training (the actual training
    # metrics are not known yet — the worker logs them separately as
    # fb_worker_train_ms). Analytics therefore lags one training event.
    if payload.query_id is not None:
        _t0 = time.perf_counter()
        complete_record(
            query_id           = payload.query_id,
            latency_ms         = payload.latency_ms,
            active_connections = payload.active_connections,
            reward             = round(reward, 4) if reward is not None else 0.0,
            ppo_step           = train_metrics.get("train_step", trainer.train_count),
            ppo_loss           = train_metrics.get("loss", train_metrics.get("avg_loss", 0.0)),
            ppo_baseline       = train_metrics.get("baseline", trainer.baseline),
            is_anomalous       = False,          # confirmed normal by forest
            anomaly_score      = anomaly_score,  # forest score (positive = normal)
        )
        analytics_ms = (time.perf_counter() - _t0) * 1000.0

    # ── Step 5: Forest refit (normal samples only) ───────────────────
    # The refit threshold is still checked here, but detector.fit() is
    # ENQUEUED to the worker; forest_refit_ms stays 0.0 at request time.
    forest_retrained, forest_refit_ms = _accumulate_forest_sample(
        wait_ms, payload.active_connections)

    feedback_total_ms = (time.perf_counter() - _fb_start) * 1000.0

    return {
        "status":           "received",
        "anomaly_detected": False,
        "anomaly_score":    anomaly_score,
        "reward":           round(reward, 4) if reward is not None else None,
        "exec_time_ms":     exec_ms,
        "wait_time_ms":     wait_ms,
        "buffer_size":      n,
        "training":         train_metrics,
        "ppo_batch_fired":  ppo_batch_fired,
        "forest_retrained": forest_retrained,
        "server_timings_ms": {
            "anomaly_predict":  round(anomaly_predict_ms,  3),
            "training":         round(training_ms,         3),
            "checkpoint_save":  round(checkpoint_save_ms,  3),
            "analytics":        round(analytics_ms,        3),
            "forest_refit":     round(forest_refit_ms,     3),
            "feedback_total":   round(feedback_total_ms,   3),
        },
    }


def _accumulate_forest_sample(wait_ms: float,
                               connections: int) -> tuple[bool, float]:
    """
    Add one [wait_ms, connections] sample to the forest buffer.
    Only NORMAL (non-anomalous) samples reach this function.
    exec_ms deliberately excluded — cannot be a DoS signal since
    exec_time can be high due to PPO's own suboptimal join order.
    Triggers refit when MIN_SAMPLES_TO_FIT reached, then every RETRAIN_EVERY_N.

    detector.fit() is now OFFLOADED to the worker: when the threshold is
    reached the sample snapshot is enqueued and this returns (False, 0.0)
    because the refit no longer runs synchronously on the request path.
    """
    sample = [wait_ms, float(connections)]
    with _forest_lock:
        _forest_samples.append(sample)
        n       = len(_forest_samples)
        samples = list(_forest_samples)

    should_fit = (n == MIN_SAMPLES_TO_FIT or
                  (n > MIN_SAMPLES_TO_FIT and (n - MIN_SAMPLES_TO_FIT) % RETRAIN_EVERY_N == 0))

    if should_fit and n >= MIN_SAMPLES_TO_FIT:
        _enqueue(("forest_refit", samples))

    return False, 0.0


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
    with _stats_lock:
        dropped       = _dropped_count
        err_count     = _worker_error_count
        last_err      = _last_worker_error
        last_train_ms = _last_worker_train_ms
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
        # Worker / offload status
        "worker_queue_depth":     _worker_queue.qsize(),
        "worker_queue_max":       QUEUE_MAX,
        "worker_dropped_count":   dropped,
        "worker_error_count":     err_count,
        "worker_last_error":      last_err,
        "worker_last_train_ms":   round(last_train_ms, 3),
    }


def get_replay_buffer() -> list:
    with _buffer_lock:
        return list(_replay_buffer)
