from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn

from rlcard_lab.rlcard_env import NUM_ACTIONS, legal_action_mask


@dataclass
class PolicyConfig:
    obs_dim: int = 54
    hidden_dim: int = 256
    value_coef: float = 0.5
    entropy_coef: float = 0.01


class TwoHeadNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # type head: fold/call/raise
        self.type_head = nn.Linear(hidden_dim, 3)
        # size head: half-pot / pot / all-in
        self.size_head = nn.Linear(hidden_dim, 3)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.trunk(obs)
        type_logits = self.type_head(feats)
        size_logits = self.size_head(feats)
        value = self.value_head(feats).squeeze(-1)
        return type_logits, size_logits, value


def combine_logits(type_logits: torch.Tensor, size_logits: torch.Tensor) -> torch.Tensor:
    # Action mapping (RLCard NLHE):
    # 0 fold, 1 check/call, 2 raise_half_pot, 3 raise_pot, 4 all_in
    action_logits = torch.zeros(type_logits.shape[0], NUM_ACTIONS, device=type_logits.device)
    action_logits[:, 0] = type_logits[:, 0]
    action_logits[:, 1] = type_logits[:, 1]
    action_logits[:, 2] = type_logits[:, 2] + size_logits[:, 0]
    action_logits[:, 3] = type_logits[:, 2] + size_logits[:, 1]
    action_logits[:, 4] = type_logits[:, 2] + size_logits[:, 2]
    return action_logits


class TwoHeadPolicy:
    def __init__(self, config: PolicyConfig, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.net = TwoHeadNet(config.obs_dim, config.hidden_dim).to(self.device)

    def act(self, state: dict, deterministic: bool = False):
        obs = torch.tensor(state["obs"], dtype=torch.float32, device=self.device).unsqueeze(0)
        mask = torch.tensor(legal_action_mask(state), dtype=torch.float32, device=self.device).unsqueeze(0)
        type_logits, size_logits, value = self.net(obs)
        logits = combine_logits(type_logits, size_logits)
        logits = logits.masked_fill(mask == 0, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        action = torch.argmax(dist.probs, dim=-1) if deterministic else dist.sample()
        logprob = dist.log_prob(action)
        return int(action.item()), logprob.squeeze(0), value.squeeze(0), dist.entropy().squeeze(0)

    def evaluate_actions(self, obs: torch.Tensor, mask: torch.Tensor, actions: torch.Tensor):
        type_logits, size_logits, value = self.net(obs)
        logits = combine_logits(type_logits, size_logits)
        logits = logits.masked_fill(mask == 0, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        logprob = dist.log_prob(actions)
        entropy = dist.entropy()
        return logprob, entropy, value

    def parameters(self):
        return self.net.parameters()

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict)


class TrajectoryBuffer:
    def __init__(self):
        self.logprobs = []
        self.values = []
        self.entropies = []
        self.returns = []

    def add(self, logprob, value, entropy, ret):
        self.logprobs.append(logprob)
        self.values.append(value)
        self.entropies.append(entropy)
        self.returns.append(ret)

    def as_tensors(self, device: torch.device):
        return (
            torch.stack(self.logprobs).to(device),
            torch.stack(self.values).to(device),
            torch.stack(self.entropies).to(device),
            torch.tensor(self.returns, dtype=torch.float32, device=device),
        )
