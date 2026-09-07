import torch
import torch.nn as nn
import torch.nn.functional as F


class ExplainableJoinOptimizer(nn.Module):
    """
    Three-layer network with a built-in attention mechanism.

    Layers:
      feature_layer  : Linear(input_dim -> hidden_dim)  -- feature extraction
      attention_layer: Linear(hidden_dim -> 1) + softmax -- XAI weights per table
      policy_head    : Linear(hidden_dim -> hidden_dim)  -- action logits

    The attention_weights returned are the explainability output:
    each value is the fraction of the model's "focus" on that table.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super(ExplainableJoinOptimizer, self).__init__()
        self.feature_layer   = nn.Linear(input_dim, hidden_dim)
        self.attention_layer = nn.Linear(hidden_dim, 1)
        self.policy_head     = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, state_vector: torch.Tensor):
        features         = F.relu(self.feature_layer(state_vector))    # [N, hidden]
        attention_scores = self.attention_layer(features)               # [N, 1]
        attention_weights = F.softmax(attention_scores, dim=0)         # [N, 1]
        context_vector   = torch.sum(attention_weights * features, dim=0)  # [hidden]
        action_logits    = self.policy_head(context_vector)             # [hidden]
        return action_logits, attention_weights


class PPOTrainer:
    """
    Policy gradient trainer for ExplainableJoinOptimizer.

    Implements REINFORCE with a running-mean baseline (advantage estimation),
    gradient norm clipping, and Adam optimisation.  This is the foundation for
    full PPO; the clipped-ratio surrogate objective can be layered on top once
    old log-probs are stored alongside each experience.

    Training signal:
      loss = -log_prob(chosen_first_table) * advantage
      advantage = reward - running_baseline

    The log-prob is taken over the attention distribution, treating the table
    chosen as the first join partner as the primary discrete action.
    """

    CLIP_EPS      = 0.2    # kept for future ratio-clipping extension
    GRAD_CLIP     = 0.5    # max gradient norm
    BASELINE_ALPHA = 0.05  # EMA smoothing factor for running baseline

    def __init__(self, model: ExplainableJoinOptimizer, lr: float = 1e-3):
        self.model     = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.baseline  = 0.0   # exponential moving average of rewards
        self.train_count = 0
        self.total_loss  = 0.0

    # ------------------------------------------------------------------
    def _compute_log_prob(self,
                          state_tensor: torch.Tensor,
                          chosen_order: list,
                          tables: list) -> torch.Tensor:
        """
        Returns log P(chosen_first_table | state) under the current policy.

        The attention distribution acts as our policy: softmax over tables.
        We pick the log-prob of whichever table ranked first in chosen_order.
        """
        _, attention_weights = self.model(state_tensor)
        probs = attention_weights.squeeze(-1)              # [N_tables]
        table_to_idx = {t: i for i, t in enumerate(tables)}
        first_idx = table_to_idx.get(chosen_order[0], 0)
        log_prob  = torch.log(probs[first_idx] + 1e-8)    # add eps for stability
        return log_prob

    # ------------------------------------------------------------------
    def train_step(self, experience: dict) -> dict:
        """
        One gradient-descent step on a single (state, action, reward) experience.

        Steps:
          1. Rebuild state tensor from table names (cache hit)
          2. Update running baseline via EMA
          3. Compute log-prob of chosen action under current policy
          4. Compute loss = -log_prob * advantage
          5. Clip gradients and step Adam

        Returns a metrics dict for logging.
        """
        from vectorizer.schema_vectorizer import build_state_tensor

        tables       = experience["tables"]
        chosen_order = experience["chosen_order"]
        reward       = experience["reward"]

        # 1. State tensor (vectorizer cache keeps this fast)
        state_tensor = build_state_tensor(tables)

        # 2. Update baseline
        self.baseline = ((1.0 - self.BASELINE_ALPHA) * self.baseline
                         + self.BASELINE_ALPHA * reward)
        advantage = reward - self.baseline

        # 3. Current policy log-prob
        log_prob = self._compute_log_prob(state_tensor, chosen_order, tables)

        # 4. Policy gradient loss (REINFORCE)
        loss = -log_prob * advantage

        # 5. Gradient update
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