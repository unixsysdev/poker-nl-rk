from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlcard_lab.rlcard_env import make_env
from rlcard_lab.rlcard_policy import PolicyConfig, TwoHeadPolicy


class RandomPolicy:
    def __init__(self, num_actions: int):
        self.num_actions = num_actions

    def act(self, state):
        legal_actions = list(state["legal_actions"].keys())
        return int(np.random.choice(legal_actions))


def load_policy(path: str, device: str):
    state = torch.load(path, map_location=device)
    config = PolicyConfig(**state["config"])
    policy = TwoHeadPolicy(config, device=device)
    policy.load_state_dict(state["model"])
    return policy


def evaluate(env, policy, opponent, episodes):
    returns = []
    for _ in range(episodes):
        state, player_id = env.reset()
        while not env.is_over():
            if player_id == 0:
                action, _, _, _ = policy.act(state, deterministic=True)
            else:
                if isinstance(opponent, TwoHeadPolicy):
                    action, _, _, _ = opponent.act(state, deterministic=True)
                else:
                    action = opponent.act(state)
            state, player_id = env.step(action)
        returns.append(env.get_payoffs()[0])
    return float(np.mean(returns))


def _lbr_rollout(env, policy, lbr_player, rng):
    steps = 0
    while not env.is_over():
        state = env.get_state(env.get_player_id())
        if env.get_player_id() == lbr_player:
            legal_actions = list(state["legal_actions"].keys())
            action = int(rng.choice(legal_actions))
        else:
            action, _, _, _ = policy.act(state, deterministic=False)
        env.step(action)
        steps += 1
    payoff = float(env.get_payoffs()[lbr_player])
    for _ in range(steps):
        env.step_back()
    return payoff, steps


def lbr_action(env, policy, lbr_player, rollouts, rng):
    state = env.get_state(env.get_player_id())
    legal_actions = list(state["legal_actions"].keys())
    if len(legal_actions) == 1:
        return int(legal_actions[0])

    best_action = int(legal_actions[0])
    best_value = -float("inf")

    for action in legal_actions:
        total = 0.0
        for _ in range(rollouts):
            env.step(int(action))
            payoff, _ = _lbr_rollout(env, policy, lbr_player, rng)
            env.step_back()
            total += payoff
        avg = total / rollouts
        if avg > best_value:
            best_value = avg
            best_action = int(action)
    return best_action


def evaluate_lbr(env, policy, episodes, rollouts, seed):
    rng = np.random.default_rng(seed)
    returns = []
    for i in range(episodes):
        state, player_id = env.reset()
        policy_player = i % 2
        lbr_player = 1 - policy_player

        while not env.is_over():
            if player_id == policy_player:
                action, _, _, _ = policy.act(state, deterministic=True)
            else:
                action = lbr_action(env, policy, lbr_player, rollouts, rng)
            state, player_id = env.step(action)

        returns.append(env.get_payoffs()[policy_player])
    return float(np.mean(returns))


def main():
    parser = argparse.ArgumentParser(description="Evaluate RLCard NLHE policies.")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--opponent", choices=["random", "policy", "lbr"], default="random")
    parser.add_argument("--opponent-policy", default=None)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lbr-rollouts", type=int, default=16)
    args = parser.parse_args()

    env = make_env(seed=args.seed, num_players=2, allow_step_back=args.opponent == "lbr")
    policy = load_policy(args.policy, device=args.device)

    if args.opponent == "policy":
        if not args.opponent_policy:
            raise ValueError("--opponent-policy is required when --opponent=policy")
        opponent = load_policy(args.opponent_policy, device=args.device)
        score = evaluate(env, policy, opponent, args.episodes)
    elif args.opponent == "lbr":
        score = evaluate_lbr(env, policy, args.episodes, args.lbr_rollouts, args.seed)
    else:
        opponent = RandomPolicy(env.num_actions)
        score = evaluate(env, policy, opponent, args.episodes)
    print(f"avg_return={score:.4f}")


if __name__ == "__main__":
    main()
