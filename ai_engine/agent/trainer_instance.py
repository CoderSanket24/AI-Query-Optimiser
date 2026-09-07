"""
trainer_instance.py
-------------------
Module-level singleton that holds the ONE model + trainer shared across the
entire FastAPI process.

Both /optimize and /feedback import `trainer` from here so they always
operate on the same weights:
  - /optimize reads  trainer.model  to produce join hints
  - /feedback writes trainer.train_step() to update those weights
"""

from agent.ppo_agent import ExplainableJoinOptimizer, PPOTrainer

# input_dim = 10  (matches the 10-feature schema vectorizer output)
# hidden_dim = 32 (network width)
_model = ExplainableJoinOptimizer(input_dim=10, hidden_dim=32)
trainer = PPOTrainer(model=_model, lr=1e-3)