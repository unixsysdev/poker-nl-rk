from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_poker_lab.poker_policy import PolicyConfig, TwoHeadPolicy
from custom_poker_lab.cuda_lab.cuda_env import CudaEnvConfig, CudaNLHEEnv


class RandomPolicy:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, mask):
        actions = [i for i, v in enumerate(mask) if v > 0]
        if not actions:
            return 1, 0.0
        return int(self.rng.choice(actions)), float(self.rng.random())


def main():
    parser = argparse.ArgumentParser(description="Evaluate CUDA NLHE policy.")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-players", type=int, default=6)
    parser.add_argument("--stack", type=int, default=20000)
    parser.add_argument("--small-blind", type=int, default=50)
    parser.add_argument("--big-blind", type=int, default=100)
    parser.add_argument("--max-raises", type=int, default=0)
    parser.add_argument("--ante", type=int, default=0)
    parser.add_argument("--rake-pct", type=float, default=0.0)
    parser.add_argument("--rake-cap", type=int, default=0)
    parser.add_argument("--rake-cap-hand", type=int, default=0)
    parser.add_argument("--rake-cap-street", type=int, default=0)
    parser.add_argument("--history-len", type=int, default=12)
    parser.add_argument("--hands-per-episode", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    state = torch.load(args.policy, map_location=args.device)
    hidden_dim = state.get("hidden_dim", 256)
    env_config = CudaEnvConfig(
        batch_size=1,
        num_players=args.num_players,
        stack=args.stack,
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        max_raises_per_round=args.max_raises,
        ante=args.ante,
        rake_pct=args.rake_pct,
        rake_cap=args.rake_cap,
        rake_cap_per_hand=args.rake_cap_hand,
        rake_cap_per_street=args.rake_cap_street,
        history_len=args.history_len,
        hands_per_episode=args.hands_per_episode,
        seed=args.seed,
        device=args.device,
    )
    env = CudaNLHEEnv(env_config)
    policy = TwoHeadPolicy(PolicyConfig(obs_dim=env.obs_dim, hidden_dim=hidden_dim), device=args.device)
    policy.load_state_dict(state["model"])
    opponent = RandomPolicy(seed=args.seed + 7)

    returns = []
    for _ in range(args.episodes):
        env.reset()
        while not bool(env.episode_over[0].item()):
            obs, mask, current = env.get_obs()
            if int(current[0].item()) == 0:
                action_type, bet_frac, *_ = policy.act(
                    {"obs": obs[0].detach().cpu().numpy(), "legal_action_mask": mask[0].detach().cpu().numpy()},
                    deterministic=True,
                )
                action_types = torch.tensor([action_type], device=env.device, dtype=torch.int64)
                bet_fracs = torch.tensor([bet_frac], device=env.device, dtype=torch.float32)
            else:
                a_type, b_frac = opponent.act(mask[0].detach().cpu().numpy())
                action_types = torch.tensor([a_type], device=env.device, dtype=torch.int64)
                bet_fracs = torch.tensor([b_frac], device=env.device, dtype=torch.float32)
            env.step(action_types, bet_fracs)
        returns.append(float(env.get_payoffs()[0, 0].item()))
    print(f"avg_return={float(np.mean(returns)):.4f}")


if __name__ == "__main__":
    main()
