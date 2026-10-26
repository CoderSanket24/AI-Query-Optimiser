import os
import time
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

_BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR  = os.path.join(_BASE_DIR, "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "ppo_checkpoint.pt")


class ExplainableJoinOptimizer(nn.Module):
    """
    Three-layer neural net with built-in attention (XAI).

      feature_layer   Linear(input_dim -> hidden_dim)  feature extraction
      attention_layer Linear(hidden_dim -> 1) + softmax  XAI weights per table
      policy_head     Linear(hidden_dim -> hidden_dim)  (kept; future value head)

    Action space
    ------------
    A join order is a permutation of the N table aliases sampled WITHOUT
    replacement from a Categorical distribution:

      1. Run forward pass → attention_weights [N]
      2. Among CONNECTED candidates, sample table i₁ ~ Categorical(sub_probs)
      3. Zero out i₁, renormalise remaining → repeat until all placed.

    "Connected" means: the candidate has at least one join edge to a table
    already in the prefix — so we never produce a cross-product prefix.
    When no candidate is connected (isolated node or first step), we fall
    back to a weighted draw among all remaining tables.

    The log-probability of the full permutation is the sum of per-step
    log-probs, analogous to pointer-network action log-probs.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super(ExplainableJoinOptimizer, self).__init__()
        self.feature_layer   = nn.Linear(input_dim, hidden_dim)
        self.attention_layer = nn.Linear(hidden_dim, 1)
        self.policy_head     = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, state_vector: torch.Tensor):
        features          = F.relu(self.feature_layer(state_vector))       # [N, hidden]
        attention_scores  = self.attention_layer(features)                  # [N, 1]
        attention_weights = F.softmax(attention_scores, dim=0)             # [N, 1]
        context_vector    = torch.sum(attention_weights * features, dim=0) # [hidden]
        action_logits     = self.policy_head(context_vector)               # [hidden]
        return action_logits, attention_weights.squeeze(-1)                 # [hidden], [N]


# ── Join-graph helpers ─────────────────────────────────────────────────────────

def _parse_join_graph(aliases: list[str], join_conditions: list[str]) -> dict[str, set[str]]:
    """
    Build an adjacency set from join_conditions.

    join_conditions: list of "leftAlias,rightAlias" strings produced by the
    Java AST walker.  An empty list means no explicit join edges were found
    (single-table query or subquery-only joins); the graph will be fully
    connected so the sampler degrades to unconstrained sampling.

    Returns: { alias -> {neighbour_alias, ...} }
    """
    adj: dict[str, set[str]] = {a: set() for a in aliases}
    for cond in join_conditions:
        parts = cond.split(",", 1)
        if len(parts) == 2:
            left, right = parts[0].strip(), parts[1].strip()
            if left in adj and right in adj:
                adj[left].add(right)
                adj[right].add(left)
    return adj


def _connected_candidates(remaining: list[str],
                           placed_set: set[str],
                           adj: dict[str, set[str]]) -> list[str]:
    """
    Return the subset of `remaining` that has at least one join-edge to a
    table already placed.  Falls back to all of `remaining` when:
      • placed_set is empty (first table — any start is fine), or
      • no remaining table is connected (e.g. genuinely disconnected query
        fragment, or no edges were parsed).
    """
    if not placed_set:
        return remaining          # first position: any table may start

    connected = [t for t in remaining if adj.get(t, set()) & placed_set]
    return connected if connected else remaining   # fallback: no isolation info


class PPOTrainer:
    """
    Part 3+7: PPO Trainer with single-step REINFORCE and true clipped-PPO batch.

    sample_action(state_tensor, aliases, join_conditions)
      → (ordered_aliases, log_prob)
      Stochastic, graph-aware: never places a table disconnected from the
      current prefix (avoids cross-product orders).

    sequence_log_prob(state_tensor, aliases, ordered_aliases)
      → torch.Tensor (scalar, differentiable)
      Used in train_batch() so gradients flow through the full sequence.

    train_step()    REINFORCE fallback
    train_batch()   True PPO2 clipped surrogate
    """

    GRAD_CLIP      = 0.5
    BASELINE_ALPHA = 0.05
    CLIP_EPS       = 0.2
    PPO_EPOCHS     = 3

    def __init__(self, model: ExplainableJoinOptimizer, lr: float = 1e-3):
        self.model       = model
        self.optimizer   = torch.optim.Adam(model.parameters(), lr=lr)
        self.baseline    = 0.0
        self.train_count = 0
        self.total_loss  = 0.0
        # Single re-entrant lock serialising model READS (sample_action,
        # the /optimize attention forward pass) against model WRITES
        # (train_step / train_batch on the feedback worker thread).
        self.lock              = threading.RLock()
        self.last_lock_wait_ms = 0.0

    # ── Stochastic graph-aware sampling ────────────────────────────────────────

    def sample_action(self, state_tensor: torch.Tensor,
                      aliases: list[str],
                      join_conditions: list[str] | None = None) -> tuple[list[str], float]:
        """
        Sample a join-order permutation WITHOUT replacement, respecting the
        join graph so every step extends a connected prefix.

        join_conditions: list of "leftAlias,rightAlias" edge strings from Java.
          None / [] → unconstrained sampling (degrades gracefully).

        Returns (ordered_aliases, log_prob_of_full_sequence).
        """
        adj = _parse_join_graph(aliases, join_conditions or [])

        # Serialise against the training worker so the forward pass never
        # reads model weights mid-update (which would corrupt log_prob_old).
        # Time the lock wait so /optimize can report it.
        _lock_t0 = time.perf_counter()
        self.lock.acquire()
        self.last_lock_wait_ms = (time.perf_counter() - _lock_t0) * 1000.0
        try:
            with torch.no_grad():
                _, probs = self.model(state_tensor)   # [N]
                probs    = probs.clone()
        finally:
            self.lock.release()

        alias_to_global = {a: i for i, a in enumerate(aliases)}
        remaining       = list(aliases)
        placed_set: set[str] = set()
        sampled_order: list[str] = []
        log_prob = 0.0

        for _ in range(len(aliases)):
            # Restrict candidates to join-graph-connected tables
            candidates = _connected_candidates(remaining, placed_set, adj)

            # Build sub-distribution over candidates
            cand_indices = [alias_to_global[a] for a in candidates]
            sub_probs    = torch.stack([probs[i] for i in cand_indices])
            sub_probs    = sub_probs / sub_probs.sum()   # renormalise

            dist    = Categorical(sub_probs)
            local_k = dist.sample()
            log_prob += dist.log_prob(local_k).item()

            chosen = candidates[local_k.item()]
            sampled_order.append(chosen)
            placed_set.add(chosen)
            remaining.remove(chosen)

        return sampled_order, log_prob

    def sequence_log_prob(self, state_tensor: torch.Tensor,
                          aliases: list[str],
                          ordered_aliases: list[str],
                          join_conditions: list[str] | None = None) -> torch.Tensor:
        """
        Compute log π(ordered_aliases | state_tensor) DIFFERENTIABLY.
        Mirrors sample_action() but keeps the computation graph intact
        so gradients flow back through the policy.
        """
        adj = _parse_join_graph(aliases, join_conditions or [])

        _, probs = self.model(state_tensor)   # [N] with gradient
        probs    = probs.clone()

        alias_to_global = {a: i for i, a in enumerate(aliases)}
        remaining       = list(aliases)
        placed_set: set[str] = set()
        log_prob = torch.tensor(0.0)

        for alias in ordered_aliases:
            candidates   = _connected_candidates(remaining, placed_set, adj)
            cand_indices = [alias_to_global[a] for a in candidates]
            sub_probs    = torch.stack([probs[i] for i in cand_indices])
            sub_probs    = sub_probs / sub_probs.sum()

            local_k  = candidates.index(alias)
            log_prob = log_prob + torch.log(sub_probs[local_k] + 1e-8)

            placed_set.add(alias)
            remaining.remove(alias)

        return log_prob

    # ── REINFORCE fallback ─────────────────────────────────────────────────────

    def train_step(self, experience: dict) -> dict:
        """
        REINFORCE — used when log_prob_old is not available.
        Trains on the full sequence log-prob so gradient covers all join steps.
        """
        from vectorizer.schema_vectorizer import build_state_tensor

        aliases         = experience["tables"]
        chosen_order    = experience["chosen_order"]
        reward          = experience["reward"]
        stat_names      = experience.get("stat_names", aliases)
        join_conditions = experience.get("join_conditions", [])

        state_tensor = build_state_tensor(stat_names)

        # Serialise against /optimize sampling on the request thread.
        with self.lock:
            self.baseline = ((1.0 - self.BASELINE_ALPHA) * self.baseline
                             + self.BASELINE_ALPHA * reward)
            advantage = reward - self.baseline

            log_prob = self.sequence_log_prob(state_tensor, aliases, chosen_order,
                                              join_conditions)
            loss = -log_prob * advantage

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.GRAD_CLIP)
            self.optimizer.step()

            self.train_count += 1
            self.total_loss  += loss.item()

            metrics = {
                "mode":       "reinforce",
                "train_step": self.train_count,
                "loss":       round(loss.item(), 6),
                "advantage":  round(float(advantage), 4),
                "baseline":   round(self.baseline, 4),
                "reward":     round(reward, 4),
            }
        print(
            f"[PPO/REINFORCE] step={self.train_count}  "
            f"loss={metrics['loss']:.4f}  adv={metrics['advantage']:.1f}  "
            f"baseline={self.baseline:.1f}  reward={reward:.1f}"
        )
        return metrics

    # ── True PPO2 batch update ─────────────────────────────────────────────────

    def train_batch(self, experiences: list) -> dict:
        """
        True PPO2 update with clipped surrogate objective.

        Each experience dict must contain:
          state_tensor    : torch.Tensor [N, input_dim]
          aliases         : list[str]  — pg_hint_plan identifiers
          chosen_order    : list[str]  — sampled permutation
          join_conditions : list[str]  — edge strings (may be [])
          log_prob_old    : float      — log π_old(full sequence | s), server-side
          reward          : float
        """
        if not experiences:
            return {}

        # Serialise against /optimize sampling on the request thread.
        with self.lock:
            total_loss  = 0.0
            total_steps = 0
            total_clip  = 0

            for epoch in range(self.PPO_EPOCHS):
                epoch_loss = 0.0
                for exp in experiences:
                    state_tensor    = exp["state_tensor"]
                    aliases         = exp["aliases"]
                    chosen_order    = exp["chosen_order"]
                    join_conditions = exp.get("join_conditions", [])
                    log_prob_old    = exp["log_prob_old"]
                    reward          = exp["reward"]

                    self.baseline = ((1.0 - self.BASELINE_ALPHA) * self.baseline
                                     + self.BASELINE_ALPHA * reward)
                    advantage = reward - self.baseline

                    log_prob_new = self.sequence_log_prob(
                        state_tensor, aliases, chosen_order, join_conditions)

                    ratio = torch.exp(
                        log_prob_new - torch.tensor(log_prob_old, dtype=torch.float32))

                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio,
                                        1.0 - self.CLIP_EPS,
                                        1.0 + self.CLIP_EPS) * advantage
                    loss  = -torch.min(surr1, surr2)

                    if (ratio.item() < 1.0 - self.CLIP_EPS or
                            ratio.item() > 1.0 + self.CLIP_EPS):
                        total_clip += 1

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.GRAD_CLIP)
                    self.optimizer.step()

                    epoch_loss   += loss.item()
                    total_loss   += loss.item()
                    total_steps  += 1
                    self.train_count += 1
                    self.total_loss  += loss.item()

                avg_epoch = epoch_loss / len(experiences)
                print(
                    f"[PPO/Clip] epoch={epoch+1}/{self.PPO_EPOCHS}  "
                    f"avg_loss={avg_epoch:.4f}  "
                    f"baseline={self.baseline:.1f}  "
                    f"clip_events={total_clip}"
                )

            avg_loss   = total_loss / max(total_steps, 1)
            avg_reward = sum(e["reward"] for e in experiences) / len(experiences)
            clip_rate  = round(total_clip / max(total_steps, 1) * 100, 1)

            metrics = {
                "mode":          "ppo_clip",
                "train_step":    self.train_count,
                "batch_size":    len(experiences),
                "ppo_epochs":    self.PPO_EPOCHS,
                "clip_eps":      self.CLIP_EPS,
                "avg_loss":      round(avg_loss, 6),
                "clip_rate_pct": clip_rate,
                "avg_reward":    round(avg_reward, 4),
                "baseline":      round(self.baseline, 4),
            }
        print(
            f"[PPO/Clip] BATCH DONE  batch={len(experiences)}  "
            f"avg_loss={avg_loss:.4f}  clip_rate={clip_rate}%  "
            f"avg_reward={avg_reward:.1f}"
        )
        return metrics

    # ── Persistence ────────────────────────────────────────────────────────────

    def save_checkpoint(self):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        torch.save({
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "baseline":             self.baseline,
            "train_count":          self.train_count,
            "total_loss":           self.total_loss,
        }, CHECKPOINT_PATH)
        print(f"[PPO] Checkpoint saved: step={self.train_count}")

    def load_checkpoint(self) -> bool:
        if not os.path.exists(CHECKPOINT_PATH):
            print("[PPO] No checkpoint found. Starting with fresh weights.")
            return False
        checkpoint = torch.load(CHECKPOINT_PATH, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.baseline    = checkpoint.get("baseline",    0.0)
        self.train_count = checkpoint.get("train_count", 0)
        self.total_loss  = checkpoint.get("total_loss",  0.0)
        print(
            f"[PPO] Checkpoint loaded: step={self.train_count}  "
            f"baseline={self.baseline:.2f}  total_loss={self.total_loss:.4f}"
        )
        return True

    @property
    def checkpoint_exists(self) -> bool:
        return os.path.exists(CHECKPOINT_PATH)

    @property
    def checkpoint_path(self) -> str:
        return CHECKPOINT_PATH