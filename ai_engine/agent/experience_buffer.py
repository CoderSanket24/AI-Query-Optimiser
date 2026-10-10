"""
experience_buffer.py
--------------------
Bridges the optimize → feedback gap for true PPO.

At /optimize time:
  store_pending(query_id, state_tensor, log_prob_old, aliases, stat_names)

At /feedback time:
  pop_pending(query_id) → dict with state_tensor + log_prob_old

Entries expire after MAX_PENDING_AGE_SEC to prevent memory leaks
from queries whose feedback never arrives.

Note: log_prob_old is the log-probability of the FULL sampled permutation
(not just the first table) and is stored server-side so the client cannot
tamper with it.
"""

import time
import threading

MAX_PENDING_AGE_SEC = 300   # discard pending entries older than 5 minutes

_pending: dict = {}          # query_id -> {state_tensor, log_prob_old, aliases, stat_names, ts}
_lock = threading.Lock()


def store_pending(query_id: int, state_tensor, log_prob_old: float,
                  aliases: list, stat_names: list) -> None:
    with _lock:
        _pending[query_id] = {
            "state_tensor": state_tensor,
            "log_prob_old": log_prob_old,   # log π_old(full sequence | s), tamper-proof
            "aliases":      aliases,         # pg_hint_plan identifiers
            "stat_names":   stat_names,      # physical names for schema stats
            "ts":           time.time(),
        }
        _evict_old()


def pop_pending(query_id: int) -> dict | None:
    with _lock:
        _evict_old()
        return _pending.pop(query_id, None)


def _evict_old() -> None:
    """Remove stale entries (called inside lock)."""
    now = time.time()
    stale = [qid for qid, v in _pending.items()
             if now - v["ts"] > MAX_PENDING_AGE_SEC]
    for qid in stale:
        del _pending[qid]
    if stale:
        print(f"[ExpBuf] Evicted {len(stale)} stale pending entries")


def pending_count() -> int:
    with _lock:
        return len(_pending)
