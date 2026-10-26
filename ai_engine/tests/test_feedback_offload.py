"""
test_feedback_offload.py
------------------------
Verifies Part 2b-1: /feedback training, checkpoint save, and forest refit
are OFFLOADED to a background worker thread (bounded queue), so the request
path stays fast.

Run from ai_engine/ inside the project venv (needs torch + scikit-learn):

    python -m unittest tests.test_feedback_offload -v

or:

    python tests/test_feedback_offload.py
"""

import os
import sys

# Make `ai_engine` importable regardless of how the test is launched.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import queue
import tempfile
import threading
import time
import unittest

DEPS_OK = True
DEPS_ERR = None
try:
    import torch
    from models.feedback_model import FeedbackPayload
    from agent.trainer_instance import trainer
    from agent.experience_buffer import store_pending
    from anomaly.detector_instance import detector
    import anomaly.isolation_forest as iforest
    import agent.ppo_agent as ppo_agent
    from routers import feedback as feedback_mod
    from routers.feedback import receive_feedback
except Exception as exc:  # pragma: no cover
    DEPS_OK = False
    DEPS_ERR = exc


def _make_state_tensor(n_tables: int = 2):
    return torch.randn((n_tables, 11))


def _make_payload(query_id, aliases, chosen_order):
    return FeedbackPayload(
        tables=aliases,
        real_names=aliases,
        chosen_order=chosen_order,
        latency_ms=60.0,
        exec_time_ms=50.0,
        wait_time_ms=1.0,
        active_connections=1,
        query_id=query_id,
    )


@unittest.skipUnless(DEPS_OK, f"missing dependency: {DEPS_ERR}")
class FeedbackOffloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Route checkpoint + forest writes to a temp dir so the test never
        # clobbers the real checkpoints/ files.
        cls._tmp = tempfile.TemporaryDirectory()
        cls._real_ckpt_path = ppo_agent.CHECKPOINT_PATH
        cls._real_forest_path = iforest.FOREST_PATH
        ppo_agent.CHECKPOINT_PATH = os.path.join(cls._tmp.name, "ppo_checkpoint.pt")
        iforest.FOREST_PATH = os.path.join(cls._tmp.name, "isolation_forest.pkl")

        # Reset the forest so predict() is deterministic (unfitted -> normal).
        detector._forest = None
        detector.is_fitted = False
        detector.n_samples_trained = 0

    @classmethod
    def tearDownClass(cls):
        ppo_agent.CHECKPOINT_PATH = cls._real_ckpt_path
        iforest.FOREST_PATH = cls._real_forest_path
        cls._tmp.cleanup()

    def setUp(self):
        # Clear per-request accumulators between tests.
        with feedback_mod._ppo_batch_lock:
            feedback_mod._ppo_batch.clear()
        with feedback_mod._forest_lock:
            feedback_mod._forest_samples.clear()

    # ── 20 fast payloads → offloaded, then fully drained ─────────────
    def test_a_twenty_payloads_offloaded_and_drained(self):
        aliases = ["a", "b"]
        chosen = ["a", "b"]
        state_tensor = _make_state_tensor()

        payloads = []
        for i in range(20):
            store_pending(
                query_id=i,
                state_tensor=state_tensor,
                log_prob_old=0.0,
                aliases=aliases,
                stat_names=aliases,
                join_conditions=["a,b"],
            )
            payloads.append(_make_payload(i, aliases, chosen))

        train_before = trainer.train_count

        async def _send_all(ps):
            out = []
            for p in ps:
                out.append(await receive_feedback(p))
            return out

        results = asyncio.run(_send_all(payloads))

        # 1. Every response is fast and shows the offloaded stages as 0.0.
        feedback_times = [r["server_timings_ms"]["feedback_total"] for r in results]
        max_t = max(feedback_times)
        self.assertLess(max_t, 500.0, f"feedback_total too slow: {max_t:.3f} ms")
        for r in results:
            t = r["server_timings_ms"]
            self.assertEqual(t["training"], 0.0, "training should be offloaded")
            self.assertEqual(t["checkpoint_save"], 0.0, "checkpoint should be offloaded")
            self.assertEqual(t["forest_refit"], 0.0, "forest refit should be offloaded")

        # 2. The two batch-firing responses (i=7, 15) report the batch queued.
        queued_modes = [r["training"]["mode"] for r in results]
        self.assertEqual(queued_modes[7], "queued")
        self.assertEqual(queued_modes[15], "queued")

        # 3. All queued items eventually process: queue drains.
        feedback_mod._worker_queue.join()  # blocks until all items are done

        # 2 batches * 8 exps * 3 PPO epochs = 48 train steps.
        self.assertEqual(
            trainer.train_count, train_before + 48,
            "train_count did not advance by exactly 48 after 2 batches",
        )
        self.assertTrue(trainer.checkpoint_exists, "checkpoint was not saved")
        # Forest refits ran off-thread (final refit saw all 20 samples).
        self.assertTrue(detector.is_fitted, "forest did not refit")
        self.assertEqual(detector.n_samples_trained, 20)

    # ── Full queue drops instead of blocking ─────────────────────────
    def test_b_full_queue_drops_nonblocking(self):
        old_q = feedback_mod._worker_queue
        try:
            tiny = queue.Queue(maxsize=1)
            tiny.put(("checkpoint", None))  # make it full
            feedback_mod._worker_queue = tiny  # _enqueue reads this global

            dropped_before = feedback_mod._dropped_count
            t0 = time.perf_counter()
            ok = feedback_mod._enqueue(("checkpoint", None))
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            self.assertFalse(ok, "enqueue should report a drop when full")
            self.assertEqual(
                feedback_mod._dropped_count, dropped_before + 1,
                "drop was not counted",
            )
            self.assertLess(elapsed_ms, 50.0, f"enqueue blocked: {elapsed_ms:.3f} ms")
        finally:
            feedback_mod._worker_queue = old_q

    # ── Concurrent sampling + training → no exceptions, no NaN ───────
    def test_c_concurrent_sampling_training_no_nan(self):
        # Make sure the worker is idle first so only these two threads contend.
        feedback_mod._worker_queue.join()

        aliases = ["a", "b", "c"]
        state_tensor = _make_state_tensor(n_tables=3)
        batch = [
            {
                "state_tensor": state_tensor,
                "aliases": aliases,
                "chosen_order": ["a", "b", "c"],
                "join_conditions": ["a,b", "b,c"],
                "log_prob_old": 0.0,
                "reward": -1.0,
            }
            for _ in range(8)
        ]

        errors = []
        stop = threading.Event()

        def train_loop():
            try:
                while not stop.is_set():
                    trainer.train_batch(batch)
            except Exception as exc:
                errors.append(exc)

        def sample_loop():
            try:
                while not stop.is_set():
                    trainer.sample_action(state_tensor, aliases, ["a,b", "b,c"])
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=train_loop, name="test-train")
        t2 = threading.Thread(target=sample_loop, name="test-sample")
        t1.start()
        t2.start()
        time.sleep(2.0)
        stop.set()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        self.assertEqual(errors, [], f"exceptions during concurrency: {errors}")
        for name, param in trainer.model.named_parameters():
            self.assertTrue(
                torch.isfinite(param).all(),
                f"NaN/Inf detected in parameter {name}",
            )


if __name__ == "__main__":
    unittest.main()
