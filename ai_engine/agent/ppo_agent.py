import os
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
      policy_head     Linear(hidden_dim -> hidden_dim)  unused in action selection
                      (kept for architecture compatibility; future: value head)

    Action space
    ------------
    A join order is a permutation of the N table aliases.  We model this as
    sampling WITHOUT replacement from a Categorical distribution:

      1. Run forward pass → attention_weights [N]  (probabilities over tables)
      2. Sample table i₁ ~ Categorical(attention_weights)
      3. Zero out i₁, renormalise → sample i₂, repeat.

    The log-probability of the full permutation is the sum of the per-step
    log-probs — analogous to pointer-network action log-probs used in
    neural combinatorial optimisation.

    This makes PPO's probability ratio r = π_new(a|s) / π_old(a|s) meaningful:
    both π_old and π_new are computed over the same full sequence.
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


class PPOTrainer:
    """
    Part 3+7: PPO Trainer with both single-step and true clipped-PPO batch update.

    train_step()   -- REINFORCE fallback (used when no log_prob_old available)
    train_batch()  -- True PPO2 with clipped surrogate objective:
                        L = min(r*adv, clip(r, 1-eps, 1+eps)*adv)
                      Runs PPO_EPOCHS epochs over the mini-batch.

    sample_action()        -- stochastic sampling of a full join-order permutation
    sequence_log_prob()    -- log π(permutation | state) used for PPO ratio

    Persistence (Part 4):
      save_checkpoint() / load_checkpoint()
    """

    GRAD_CLIP      = 0.5
    BASELINE_ALPHA = 0.05
    CLIP_EPS       = 0.2    # PPO clip epsilon (standard value)
    PPO_EPOCHS     = 3      # epochs per batch update

    def __init__(self, model: ExplainableJoinOptimizer, lr: float = 1e-3):
        self.model       = model
        self.optimizer   = torch.optim.Adam(model.parameters(), lr=lr)
        self.baseline    = 0.0
        self.train_count = 0
        self.total_loss  = 0.0

    # ------------------------------------------------------------------
    # Stochastic action: sample a full permutation, return its log-prob
    # ------------------------------------------------------------------

    def sample_action(self, state_tensor: torch.Tensor, aliases: list) -> tuple[list, float]:
        """
        Sample a join-order permutation WITHOUT replacement from the attention
        distribution.  Returns (ordered_aliases, log_prob_of_full_sequence).

        Called at /optimize time; no gradient tracking needed.
        """
        with torch.no_grad():
            _, probs = self.model(state_tensor)   # [N]
            probs    = probs.clone()

        n = len(aliases)
        remaining_indices  = list(range(n))
        sampled_order      = []
        log_prob           = 0.0

        for _ in range(n):
            # Extract probabilities for remaining tables and renormalise
            sub_probs = torch.stack([probs[i] for i in remaining_indices])
            sub_probs = sub_probs / sub_probs.sum()          # renormalise

            dist     = Categorical(sub_probs)
            local_k  = dist.sample()                         # index into sub_probs
            log_prob += dist.log_prob(local_k).item()

            chosen_global = remaining_indices[local_k.item()]
            sampled_order.append(aliases[chosen_global])
            remaining_indices.remove(chosen_global)

        return sampled_order, log_prob

    def sequence_log_prob(self, state_tensor: torch.Tensor,
                          aliases: list, ordered_aliases: list) -> torch.Tensor:
        """
        Compute log π(ordered_aliases | state_tensor) DIFFERENTIABLY.

        Used in train_batch() so gradients flow back through the policy.
        Mirrors the sampling logic but keeps the computation graph intact.
        """
        _, probs = self.model(state_tensor)   # [N] — with gradient
        probs    = probs.clone()

        alias_to_idx   = {a: i for i, a in enumerate(aliases)}
        remaining_idx  = list(range(len(aliases)))
        log_prob       = torch.tensor(0.0)

        for alias in ordered_aliases:
            global_idx = alias_to_idx[alias]
            sub_probs  = torch.stack([probs[i] for i in remaining_idx])
            sub_probs  = sub_probs / sub_probs.sum()   # renormalise (differentiable)

            local_k = remaining_idx.index(global_idx)
            log_prob = log_prob + torch.log(sub_probs[local_k] + 1e-8)
            remaining_idx.remove(global_idx)

        return log_prob

    # ------------------------------------------------------------------
    # REINFORCE fallback (no log_prob_old available)
    # ------------------------------------------------------------------

    def train_step(self, experience: dict) -> dict:
        """
        REINFORCE fallback — used when log_prob_old is not available.
        Trains on the full sequence log-prob so gradient signal is meaningful.
        """
        from vectorizer.schema_vectorizer import build_state_tensor

        aliases      = experience["tables"]
        chosen_order = experience["chosen_order"]
        reward       = experience["reward"]

        stat_names   = experience.get("stat_names", aliases)
        state_tensor = build_state_tensor(stat_names)

        self.baseline = ((1.0 - self.BASELINE_ALPHA) * self.baseline
                         + self.BASELINE_ALPHA * reward)
        advantage = reward - self.baseline

        log_prob = self.sequence_log_prob(state_tensor, aliases, chosen_order)
        loss     = -log_prob * advantage

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

    # ------------------------------------------------------------------
    # True PPO2 batch update
    # ------------------------------------------------------------------

    def train_batch(self, experiences: list) -> dict:
        """
        True PPO2 update with clipped surrogate objective.

        Each experience dict must contain:
          state_tensor  : torch.Tensor [N, input_dim]
          aliases       : list[str]  — pg_hint_plan identifiers (in state order)
          chosen_order  : list[str]  — sampled permutation recorded at optimize time
          log_prob_old  : float      — log π_old(full sequence | s), from pending store
          reward        : float

        Runs PPO_EPOCHS epochs over the batch.
        """
        if not experiences:
            return {}

        total_loss   = 0.0
        total_steps  = 0
        total_clip   = 0

        for epoch in range(self.PPO_EPOCHS):
            epoch_loss = 0.0
            for exp in experiences:
                state_tensor  = exp["state_tensor"]
                aliases       = exp["aliases"]
                chosen_order  = exp["chosen_order"]
                log_prob_old  = exp["log_prob_old"]
                reward        = exp["reward"]

                # Update EMA baseline
                self.baseline = ((1.0 - self.BASELINE_ALPHA) * self.baseline
                                 + self.BASELINE_ALPHA * reward)
                advantage = reward - self.baseline

                # New log-prob of the same FULL SEQUENCE under current policy
                log_prob_new = self.sequence_log_prob(state_tensor, aliases, chosen_order)

                # PPO probability ratio
                ratio = torch.exp(log_prob_new - torch.tensor(log_prob_old, dtype=torch.float32))

                # Clipped surrogate
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.CLIP_EPS, 1.0 + self.CLIP_EPS) * advantage
                loss  = -torch.min(surr1, surr2)

                was_clipped = (ratio.item() < 1.0 - self.CLIP_EPS or
                               ratio.item() > 1.0 + self.CLIP_EPS)
                if was_clipped:
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

    # ------------------------------------------------------------------
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