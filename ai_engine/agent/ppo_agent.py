import os
import torch
import torch.nn as nn
import torch.nn.functional as F

_BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR  = os.path.join(_BASE_DIR, "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "ppo_checkpoint.pt")


class ExplainableJoinOptimizer(nn.Module):
    """
    Three-layer neural net with built-in attention (XAI).

      feature_layer   Linear(input_dim -> hidden_dim)  feature extraction
      attention_layer Linear(hidden_dim -> 1) + softmax  XAI weights per table
      policy_head     Linear(hidden_dim -> hidden_dim)  action logits
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
        return action_logits, attention_weights


class PPOTrainer:
    """
    Part 3+7: PPO Trainer with both single-step and true clipped-PPO batch update.

    train_step()   -- REINFORCE fallback (used when no log_prob_old available)
    train_batch()  -- True PPO2 with clipped surrogate objective:
                        L = min(r*adv, clip(r, 1-eps, 1+eps)*adv)
                      Runs PPO_EPOCHS epochs over the mini-batch.

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
    def _first_table_idx(self, tables, chosen_order) -> int:
        table_to_idx = {t: i for i, t in enumerate(tables)}
        return table_to_idx.get(chosen_order[0], 0)

    def _compute_log_prob(self, state_tensor, first_idx: int) -> torch.Tensor:
        _, attention_weights = self.model(state_tensor)
        probs = attention_weights.squeeze(-1)
        return torch.log(probs[first_idx] + 1e-8)

    # ------------------------------------------------------------------
    def compute_action_log_prob(self, state_tensor, tables, chosen_order):
        """
        Called from /optimize to get log_prob_old BEFORE training.
        Returns (log_prob_value: float, first_table_idx: int).
        No gradient tracking needed here.
        """
        with torch.no_grad():
            first_idx = self._first_table_idx(tables, chosen_order)
            log_prob  = self._compute_log_prob(state_tensor, first_idx)
        return log_prob.item(), first_idx

    # ------------------------------------------------------------------
    def train_step(self, experience: dict) -> dict:
        """
        REINFORCE fallback — used when log_prob_old is not available
        (e.g. anomaly-detected queries that were not PPO-optimized).
        """
        from vectorizer.schema_vectorizer import build_state_tensor

        tables       = experience["tables"]
        chosen_order = experience["chosen_order"]
        reward       = experience["reward"]

        state_tensor = build_state_tensor(tables)

        self.baseline = ((1.0 - self.BASELINE_ALPHA) * self.baseline
                         + self.BASELINE_ALPHA * reward)
        advantage = reward - self.baseline

        first_idx = self._first_table_idx(tables, chosen_order)
        log_prob  = self._compute_log_prob(state_tensor, first_idx)
        loss      = -log_prob * advantage

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
    def train_batch(self, experiences: list) -> dict:
        """
        True PPO2 update with clipped surrogate objective.

        Each experience dict must contain:
          state_tensor  : torch.Tensor [N, 10]
          first_idx     : int          index of the first table in chosen_order
          log_prob_old  : float        log π_old(a|s) recorded at optimize time
          reward        : float

        Runs PPO_EPOCHS epochs over the batch.
        """
        if not experiences:
            return {}

        total_loss   = 0.0
        total_steps  = 0
        total_clip   = 0     # how many times the ratio was clipped

        for epoch in range(self.PPO_EPOCHS):
            epoch_loss = 0.0
            for exp in experiences:
                state_tensor  = exp["state_tensor"]
                first_idx     = exp["first_idx"]
                log_prob_old  = exp["log_prob_old"]
                reward        = exp["reward"]

                # Update EMA baseline
                self.baseline = ((1.0 - self.BASELINE_ALPHA) * self.baseline
                                 + self.BASELINE_ALPHA * reward)
                advantage = reward - self.baseline

                # New log-prob under current policy
                log_prob_new = self._compute_log_prob(state_tensor, first_idx)

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
            "mode":        "ppo_clip",
            "train_step":  self.train_count,
            "batch_size":  len(experiences),
            "ppo_epochs":  self.PPO_EPOCHS,
            "clip_eps":    self.CLIP_EPS,
            "avg_loss":    round(avg_loss, 6),
            "clip_rate_pct": clip_rate,
            "avg_reward":  round(avg_reward, 4),
            "baseline":    round(self.baseline, 4),
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