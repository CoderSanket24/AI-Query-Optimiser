import torch
import torch.nn as nn
import torch.nn.functional as F

class ExplainableJoinOptimizer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ExplainableJoinOptimizer, self).__init__()
        
        # 1. Feature extraction layer
        self.feature_layer = nn.Linear(input_dim, hidden_dim)
        
        # 2. The Attention Mechanism (This provides our XAI)
        # It learns to assign a "weight" or "focus" percentage to each table
        self.attention_layer = nn.Linear(hidden_dim, 1)
        
        # 3. The PPO Policy Output (Predicts the best action)
        self.policy_head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, state_vector):
        # Pass state through the feature layer
        features = F.relu(self.feature_layer(state_vector))
        
        # Calculate Attention Weights (Explainability)
        attention_scores = self.attention_layer(features)
        attention_weights = F.softmax(attention_scores, dim=0)
        
        # Apply attention to the features
        context_vector = torch.sum(attention_weights * features, dim=0)
        
        # Generate the final action logits (Join order prediction)
        action_logits = self.policy_head(context_vector)
        
        return action_logits, attention_weights