from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(obs)
        logits = self.policy_head(hidden)
        value = self.value_head(hidden).squeeze(-1)
        return logits, value
