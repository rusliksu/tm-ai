import torch
import torch.nn as nn
from typing import Tuple


class PolicyValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden_sizes: list[int], action_space_size: int):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(0.1)]
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(in_dim, action_space_size)
        self.value_head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x:    (batch, state_dim) float32
        mask: (batch, action_space_size) bool — True for valid actions
        Returns logits (batch, action_space_size) and value (batch,).
        """
        features = self.backbone(x)
        logits = self.policy_head(features)
        logits = logits.masked_fill(~mask, float("-inf"))
        value = self.value_head(features).squeeze(-1)
        return logits, value
