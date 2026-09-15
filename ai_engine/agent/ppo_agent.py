import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Checkpoint stored at ai_engine/checkpoints/ppo_checkpoint.pt
_BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR  = os.path.join(_BASE_DIR, "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "ppo_checkpoint.pt")


class ExplainableJoinOptimizer(nn.Module):
    """
    Three-layer neural network with built-in attention (XAI).

    Layers:
      feature_layer   Linear(input_dim -> hidden_dim)  feature extraction
      attention_layer Linear(hidden_dim -> 1) + softmax XAI weights per table
      policy_head     Linear(hidden_dim -> hidden_dim)  action logits
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super(ExplainableJoinOptimizer, self).__init__()
        self.feature_layer   = nn.Linear(input_dim, hidden_dim)
        self.attention_layer = nn.Linear(hidden_dim, 1)
        self.policy_head     = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, state_vector: torch.Tensor):
        features          = F.relu(self.feature_layer(state_vector))   # [N, hidden]
        attention_scores  = self.attention_layer(features)              # [N, 1]
        attention_weights = F.softmax(attention_scores, dim=0)         # [N, 1]
        context_vector    = torch.sum(attention_weights * features, dim=0)  # [hidden]
        action_logits     = self.policy_head(context_vector)           # [hidden]
        return action_logits, attention_weights


class PPOTrainer:
    """
    Policy gradient trainer (REINFORCE with EMA baseline).

    Each train_step():
      1. Rebuild state tensor from table names (cache hit)
      2. Update running baseline via EMA
      3. Compute log-prob of chosen first-join table under current policy
      4. loss = -log_prob * advantage    (advantage = reward - baseline)
      5. Clip gradients, step Adam

    Persistence (Part 4):
      save_checkpoint() -- called after every train_step
      load_checkpoint() -- called at process startup from trainer_instance.py
    """

    GRAD_CLIP      = 0.5
    BASELINE_ALPHA = 0.05

    def __init__(self, model: ExplainableJoinOptimizer, lr: float = 1e-3):
        self.model       = model
        self.optimizer   = torch.optim.Adam(model.parameters(), lr=lr)
        self.baseline    = 0.0
        self.train_count = 0
        self.total_loss  = 0.0

    # ------------------------------------------------------------------
    def _compute_log_prob(self, state_tensor, chosen_order, tables):
        _, attention_weights = self.model(state_tensor)
        probs         = attention_weights.squeeze(-1)
        table_to_idx  = {t: i for i, t in enumerate(tables)}
        first_idx     = table_to_idx.get(chosen_order[0], 0)
        return torch.log(probs[first_idx] + 1e-8)

    # ------------------------------------------------------------------
    def train_step(self, experience: dict) -> dict:
        from vectorizer.schema_vectorizer import build_state_tensor

        tables       = experience["tables"]
        chosen_order = experience["chosen_order"]
        reward       = experience["reward"]

        state_tensor = build_state_tensor(tables)

        self.baseline = ((1.0 - self.BASELINE_ALPHA) * self.baseline
                         + self.BASELINE_ALPHA * reward)
        advantage = reward - self.baseline

        log_prob = self._compute_log_prob(state_tensor, chosen_order, tables)
        loss     = -log_prob * advantage

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.GRAD_CLIP)
        self.optimizer.step()

        self.train_count += 1
        self.total_loss  += loss.item()

        metrics = {
            "train_step": self.train_count,
            "loss":       round(loss.item(), 6),
            "advantage":  round(float(advantage), 4),
            "baseline":   round(self.baseline, 4),
            "reward":     round(reward, 4),
        }
        print(
            f"[PPO] step={metrics['train_step']}  "
            f"loss={metrics['loss']:.4f}  "
            f"adv={metrics['advantage']:.1f}  "
            f"baseline={metrics['baseline']:.1f}  "
            f"reward={metrics['reward']:.1f}"
        )
        return metrics

    # ------------------------------------------------------------------
    def save_checkpoint(self):
        """Persist model weights, optimizer state, and trainer scalars to disk."""
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        torch.save({
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "baseline":             self.baseline,
            "train_count":          self.train_count,
            "total_loss":           self.total_loss,
        }, CHECKPOINT_PATH)
        print(f"[PPO] Checkpoint saved: step={self.train_count}  path={CHECKPOINT_PATH}")

    # ------------------------------------------------------------------
    def load_checkpoint(self) -> bool:
        """
        Load a previously saved checkpoint on startup.
        Returns True if a checkpoint was found and loaded, False otherwise.
        """
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
            f"[PPO] Checkpoint loaded: "
            f"step={self.train_count}  "
            f"baseline={self.baseline:.2f}  "
            f"total_loss={self.total_loss:.4f}"
        )
        return True

    # ------------------------------------------------------------------
    @property
    def checkpoint_exists(self) -> bool:
        return os.path.exists(CHECKPOINT_PATH)

    @property
    def checkpoint_path(self) -> str:
        return CHECKPOINT_PATH