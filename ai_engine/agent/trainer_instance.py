"""
trainer_instance.py
-------------------
Module-level singleton: one model + one trainer shared across all requests.

On import, automatically tries to resume from the latest checkpoint.
If no checkpoint exists, starts with fresh random weights.
"""

from agent.ppo_agent import ExplainableJoinOptimizer, PPOTrainer

_model  = ExplainableJoinOptimizer(input_dim=10, hidden_dim=32)
trainer = PPOTrainer(model=_model, lr=1e-3)

# Resume learning from last session if a checkpoint exists
trainer.load_checkpoint()