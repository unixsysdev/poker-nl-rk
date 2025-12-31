from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn


@dataclass
class PolicyConfig:
    obs_dim: int
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
        self.type_head = nn.Linear(hidden_dim, 3)
        self.size_head = nn.Linear(hidden_dim, 2)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.trunk(obs)
        type_logits = self.type_head(feats)
        size_params = self.size_head(feats)
        value = self.value_head(feats).squeeze(-1)
        return type_logits, size_params, value


class TwoHeadPolicy:
    def __init__(self, config: PolicyConfig, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.net = TwoHeadNet(config.obs_dim, config.hidden_dim).to(self.device)

    def act(self, state: dict, deterministic: bool = False):
        obs = torch.tensor(state["obs"], dtype=torch.float32, device=self.device).unsqueeze(0)
        mask = torch.tensor(state["legal_action_mask"], dtype=torch.float32, device=self.device).unsqueeze(0)
        type_logits, size_params, value = self.net(obs)
        type_logits = type_logits.masked_fill(mask == 0, -1e9)
        type_dist = torch.distributions.Categorical(logits=type_logits)
        action_type = torch.argmax(type_dist.probs, dim=-1) if deterministic else type_dist.sample()
        logprob_type = type_dist.log_prob(action_type)
        entropy_type = type_dist.entropy()

        alpha = torch.nn.functional.softplus(size_params[:, 0]) + 1.0
        beta = torch.nn.functional.softplus(size_params[:, 1]) + 1.0
        size_dist = torch.distributions.Beta(alpha, beta)
        bet_frac = size_dist.mean if deterministic else size_dist.sample()
        bet_frac = bet_frac.clamp(0.0, 1.0)
        logprob_size = size_dist.log_prob(bet_frac)
        entropy_size = size_dist.entropy()

        raise_mask = (action_type == 2).float()
        logprob = logprob_type + raise_mask * logprob_size
        entropy = entropy_type + raise_mask * entropy_size

        return (
            int(action_type.item()),
            float(bet_frac.item()),
            logprob.squeeze(0),
            value.squeeze(0),
            entropy.squeeze(0),
            raise_mask.squeeze(0),
        )

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        action_type: torch.Tensor,
        bet_frac: torch.Tensor,
        raise_mask: torch.Tensor,
    ):
        type_logits, size_params, value = self.net(obs)
        type_logits = type_logits.masked_fill(mask == 0, -1e9)
        type_dist = torch.distributions.Categorical(logits=type_logits)
        logprob_type = type_dist.log_prob(action_type)
        entropy_type = type_dist.entropy()

        alpha = torch.nn.functional.softplus(size_params[:, 0]) + 1.0
        beta = torch.nn.functional.softplus(size_params[:, 1]) + 1.0
        size_dist = torch.distributions.Beta(alpha, beta)
        bet_frac = bet_frac.clamp(1e-6, 1.0 - 1e-6)
        logprob_size = size_dist.log_prob(bet_frac)
        entropy_size = size_dist.entropy()

        logprob = logprob_type + raise_mask * logprob_size
        entropy = entropy_type + raise_mask * entropy_size
        return logprob, entropy, value

    def parameters(self):
        return self.net.parameters()

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict)


class TrajectoryBuffer:
    def __init__(self):
        self.obs = []
        self.masks = []
        self.action_types = []
        self.bet_fracs = []
        self.logprobs = []
        self.values = []
        self.entropies = []
        self.raise_masks = []
        self.returns = []

    def add(self, obs, mask, action_type, bet_frac, logprob, value, entropy, raise_mask, ret):
        if isinstance(logprob, torch.Tensor):
            logprob = logprob.detach().cpu()
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
        if isinstance(entropy, torch.Tensor):
            entropy = entropy.detach().cpu()
        self.obs.append(obs)
        self.masks.append(mask)
        self.action_types.append(action_type)
        self.bet_fracs.append(bet_frac)
        self.logprobs.append(logprob)
        self.values.append(value)
        self.entropies.append(entropy)
        self.raise_masks.append(raise_mask)
        self.returns.append(ret)

    def as_tensors(self, device: torch.device):
        return (
            torch.tensor(np.array(self.obs), dtype=torch.float32, device=device),
            torch.tensor(np.array(self.masks), dtype=torch.float32, device=device),
            torch.tensor(self.action_types, dtype=torch.int64, device=device),
            torch.tensor(self.bet_fracs, dtype=torch.float32, device=device),
            torch.stack(self.logprobs).to(device),
            torch.stack(self.values).to(device),
            torch.stack(self.entropies).to(device),
            torch.tensor(self.raise_masks, dtype=torch.float32, device=device),
            torch.tensor(self.returns, dtype=torch.float32, device=device),
        )
