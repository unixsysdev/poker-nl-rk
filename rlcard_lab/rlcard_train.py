from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import torch
from torch import optim

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlcard_lab.rlcard_env import make_env
from rlcard_lab.rlcard_policy import PolicyConfig, TwoHeadPolicy, TrajectoryBuffer


def run_episode(env, policy_a, policy_b, device):
    buffers = [TrajectoryBuffer(), TrajectoryBuffer()]
    state, player_id = env.reset()

    while not env.is_over():
        if player_id == 0:
            action, logprob, value, entropy = policy_a.act(state, deterministic=False)
        else:
            action, logprob, value, entropy = policy_b.act(state, deterministic=False)

        buffers[player_id].add(logprob, value, entropy, ret=0.0)
        state, player_id = env.step(action)

    payoffs = env.get_payoffs()
    for pid in range(2):
        payoff = float(payoffs[pid])
        buffers[pid].returns = [payoff] * len(buffers[pid].returns)

    return buffers, payoffs


def evaluate(env, policy, episodes):
    returns = []
    for _ in range(episodes):
        state, player_id = env.reset()
        while not env.is_over():
            action, _, _, _ = policy.act(state, deterministic=True)
            state, player_id = env.step(action)
        payoffs = env.get_payoffs()
        returns.append(payoffs[0])
    return float(np.mean(returns))


def main():
    parser = argparse.ArgumentParser(description="Train RLCard NLHE with a two-head policy.")
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--update-every", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--save-dir", default="experiments/rlcard_nlhe")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = make_env(seed=args.seed, num_players=2)
    config = PolicyConfig(
        hidden_dim=args.hidden_dim,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
    )
    policy = TwoHeadPolicy(config, device=args.device)
    opponent = TwoHeadPolicy(config, device=args.device)
    opponent.load_state_dict(policy.state_dict())

    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    start = time.time()

    total_episodes = 0
    if args.resume:
        state = torch.load(args.resume, map_location=args.device)
        policy.load_state_dict(state["model"])
        opponent.load_state_dict(state["model"])
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        total_episodes = int(state.get("episodes", 0))
        print(f"resume_from={args.resume} episodes={total_episodes}")

    while total_episodes < args.episodes:
        batch_buffers = [TrajectoryBuffer(), TrajectoryBuffer()]
        batch_payoffs = []

        for _ in range(args.update_every):
            buffers, payoffs = run_episode(env, policy, opponent, args.device)
            batch_payoffs.append(payoffs[0])
            for pid in range(2):
                batch_buffers[pid].logprobs.extend(buffers[pid].logprobs)
                batch_buffers[pid].values.extend(buffers[pid].values)
                batch_buffers[pid].entropies.extend(buffers[pid].entropies)
                batch_buffers[pid].returns.extend(buffers[pid].returns)
            total_episodes += 1
            if total_episodes >= args.episodes:
                break

        merged = TrajectoryBuffer()
        for pid in range(2):
            merged.logprobs.extend(batch_buffers[pid].logprobs)
            merged.values.extend(batch_buffers[pid].values)
            merged.entropies.extend(batch_buffers[pid].entropies)
            merged.returns.extend(batch_buffers[pid].returns)

        logprobs, values, entropies, returns = merged.as_tensors(policy.device)
        advantages = returns - values.detach()

        policy_loss = -(logprobs * advantages).mean()
        value_loss = torch.mean((values - returns) ** 2)
        entropy_loss = -entropies.mean()

        loss = policy_loss + args.value_coef * value_loss + args.entropy_coef * entropy_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        opponent.load_state_dict(policy.state_dict())

        if args.log_every and total_episodes % args.log_every == 0:
            elapsed = time.time() - start
            avg_payoff = float(np.mean(batch_payoffs))
            print(
                f"episodes={total_episodes} avg_payoff={avg_payoff:.4f} "
                f"loss={loss.item():.4f} elapsed={elapsed:.1f}s"
            )

        if args.eval_every and total_episodes % args.eval_every == 0:
            score = evaluate(env, policy, args.eval_episodes)
            print(f"eval@{total_episodes} avg_return={score:.4f}")

        if args.save_every and total_episodes % args.save_every == 0:
            save_dir = pathlib.Path(args.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            path = save_dir / f"policy_ep_{total_episodes:06d}.pt"
            torch.save(
                {
                    "model": policy.state_dict(),
                    "config": config.__dict__,
                    "episodes": total_episodes,
                    "optimizer": optimizer.state_dict(),
                },
                path,
            )
            print(f"checkpoint_saved={path}")


if __name__ == "__main__":
    main()
