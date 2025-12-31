from __future__ import annotations

import argparse
import copy
import pathlib
import random
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn, optim

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlcard_lab.rlcard_env import legal_action_mask, make_env
from rlcard_lab.rlcard_policy import PolicyConfig, TwoHeadPolicy


@dataclass
class Rollout:
    obs: list
    mask: list
    actions: list
    logprobs: list
    values: list
    returns: list


def run_episode(env, policy, opponent, train_player, device):
    obs_list = []
    mask_list = []
    actions_list = []
    logprob_list = []
    value_list = []

    state, player_id = env.reset()
    while not env.is_over():
        if player_id == train_player:
            action, logprob, value, _ = policy.act(state, deterministic=False)
            obs_list.append(state["obs"])
            mask_list.append(legal_action_mask(state))
            actions_list.append(action)
            logprob_list.append(logprob)
            value_list.append(value)
        else:
            action, _, _, _ = opponent.act(state, deterministic=False)
        state, player_id = env.step(action)

    payoff = float(env.get_payoffs()[train_player])
    returns = [payoff] * len(actions_list)
    return Rollout(
        obs=obs_list,
        mask=mask_list,
        actions=actions_list,
        logprobs=logprob_list,
        values=value_list,
        returns=returns,
    )


def evaluate(env, policy, episodes):
    returns = []
    for _ in range(episodes):
        state, player_id = env.reset()
        while not env.is_over():
            action, _, _, _ = policy.act(state, deterministic=True)
            state, player_id = env.step(action)
        returns.append(env.get_payoffs()[0])
    return float(np.mean(returns))


def collect_rollouts(env, policy, opponent, episodes, device):
    obs = []
    masks = []
    actions = []
    logprobs = []
    values = []
    returns = []

    for _ in range(episodes):
        train_player = random.randint(0, 1)
        rollout = run_episode(env, policy, opponent, train_player, device)
        obs.extend(rollout.obs)
        masks.extend(rollout.mask)
        actions.extend(rollout.actions)
        logprobs.extend(rollout.logprobs)
        values.extend(rollout.values)
        returns.extend(rollout.returns)

    obs = torch.tensor(np.array(obs), dtype=torch.float32, device=device)
    masks = torch.tensor(np.array(masks), dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.int64, device=device)
    logprobs = torch.stack(logprobs).detach().to(device)
    values = torch.stack(values).detach().to(device)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    return obs, masks, actions, logprobs, values, returns


def ppo_update(policy, optimizer, batch, clip_ratio, value_coef, entropy_coef, epochs, minibatch):
    obs, masks, actions, old_logprobs, values, returns = batch
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = obs.shape[0]
    idx = torch.randperm(n, device=obs.device)
    for _ in range(epochs):
        for start in range(0, n, minibatch):
            end = start + minibatch
            mb_idx = idx[start:end]
            mb_obs = obs[mb_idx]
            mb_masks = masks[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_logprobs = old_logprobs[mb_idx]
            mb_returns = returns[mb_idx]
            mb_adv = advantages[mb_idx]

            logprob, entropy, value = policy.evaluate_actions(mb_obs, mb_masks, mb_actions)
            ratio = torch.exp(logprob - mb_old_logprobs)
            clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
            policy_loss = -(torch.min(ratio * mb_adv, clipped * mb_adv)).mean()
            value_loss = nn.functional.mse_loss(value, mb_returns)
            entropy_loss = -entropy.mean()

            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def main():
    parser = argparse.ArgumentParser(description="PPO training on RLCard NLHE.")
    parser.add_argument("--episodes", type=int, default=200000)
    parser.add_argument("--rollout-episodes", type=int, default=200)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=1024)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=20000)
    parser.add_argument("--eval-episodes", type=int, default=2000)
    parser.add_argument("--save-every", type=int, default=20000)
    parser.add_argument("--save-dir", default="experiments/rlcard_nlhe_ppo")
    parser.add_argument("--resume", default=None)

    parser.add_argument("--pool-size", type=int, default=8)
    parser.add_argument("--pool-add-every", type=int, default=20000)
    parser.add_argument("--pool-prob", type=float, default=0.5)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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
    pool = []

    if args.resume:
        state = torch.load(args.resume, map_location=args.device)
        policy.load_state_dict(state["model"])
        opponent.load_state_dict(state["model"])
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        total_episodes = int(state.get("episodes", 0))
        pool = state.get("pool", [])
        print(f"resume_from={args.resume} episodes={total_episodes}")

    while total_episodes < args.episodes:
        if pool and random.random() < args.pool_prob:
            opponent.load_state_dict(random.choice(pool))
        else:
            opponent.load_state_dict(policy.state_dict())

        batch = collect_rollouts(env, policy, opponent, args.rollout_episodes, args.device)
        total_episodes += args.rollout_episodes

        ppo_update(
            policy,
            optimizer,
            batch,
            clip_ratio=args.clip_ratio,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            epochs=args.ppo_epochs,
            minibatch=args.minibatch,
        )

        if args.pool_add_every and total_episodes % args.pool_add_every == 0:
            pool.append(copy.deepcopy(policy.state_dict()))
            if len(pool) > args.pool_size:
                pool.pop(0)

        if args.log_every and total_episodes % args.log_every == 0:
            elapsed = time.time() - start
            print(f"episodes={total_episodes} elapsed={elapsed:.1f}s pool={len(pool)}")

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
                    "pool": pool,
                },
                path,
            )
            print(f"checkpoint_saved={path}")


if __name__ == "__main__":
    main()
