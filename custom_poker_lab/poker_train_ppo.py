from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import pathlib
import random
import sys
import time

import numpy as np
import torch
from torch import nn, optim

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataclasses import replace

from custom_poker_lab.poker_env import EnvConfig, NoLimitHoldemEnv
from custom_poker_lab.poker_policy import PolicyConfig, TrajectoryBuffer, TwoHeadPolicy
from custom_poker_lab import poker_eval as evaler


def run_episode(env: NoLimitHoldemEnv, policy: TwoHeadPolicy):
    buffers = [TrajectoryBuffer() for _ in range(env.config.num_players)]
    state, player_id = env.reset()

    while not env.is_over():
        action_type, bet_frac, logprob, value, entropy, raise_mask = policy.act(state)
        buffers[player_id].add(
            state["obs"],
            state["legal_action_mask"],
            action_type,
            bet_frac,
            logprob,
            value,
            entropy,
            float(raise_mask.item()),
            0.0,
        )
        state, player_id = env.step(action_type, bet_frac)

    payoffs = env.get_payoffs()
    for pid in range(env.config.num_players):
        buffers[pid].returns = [float(payoffs[pid])] * len(buffers[pid].returns)
    return buffers


def collect_rollouts(policy_state, env_config, episodes, seed, hidden_dim):
    policy = TwoHeadPolicy(
        PolicyConfig(obs_dim=env_config_obs_dim(env_config), hidden_dim=hidden_dim),
        device="cpu",
    )
    policy.load_state_dict(policy_state)
    local_config = replace(env_config, seed=seed)
    env = NoLimitHoldemEnv(local_config)

    merged = TrajectoryBuffer()
    for _ in range(episodes):
        buffers = run_episode(env, policy)
        for buf in buffers:
            merged.obs.extend(buf.obs)
            merged.masks.extend(buf.masks)
            merged.action_types.extend(buf.action_types)
            merged.bet_fracs.extend(buf.bet_fracs)
            merged.logprobs.extend(buf.logprobs)
            merged.values.extend(buf.values)
            merged.entropies.extend(buf.entropies)
            merged.raise_masks.extend(buf.raise_masks)
            merged.returns.extend(buf.returns)
    return merged


def merge_buffers(buffers):
    merged = TrajectoryBuffer()
    for buf in buffers:
        merged.obs.extend(buf.obs)
        merged.masks.extend(buf.masks)
        merged.action_types.extend(buf.action_types)
        merged.bet_fracs.extend(buf.bet_fracs)
        merged.logprobs.extend(buf.logprobs)
        merged.values.extend(buf.values)
        merged.entropies.extend(buf.entropies)
        merged.raise_masks.extend(buf.raise_masks)
        merged.returns.extend(buf.returns)
    return merged


def env_config_obs_dim(env_config: EnvConfig) -> int:
    return 52 + 5 + 7 * env_config.num_players + 4 * env_config.history_len


def ppo_update(policy, optimizer, batch, clip_ratio, value_coef, entropy_coef, epochs, minibatch):
    (
        obs,
        masks,
        action_types,
        bet_fracs,
        old_logprobs,
        values,
        entropies,
        raise_masks,
        returns,
    ) = batch
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
            mb_actions = action_types[mb_idx]
            mb_bet_fracs = bet_fracs[mb_idx]
            mb_old_logprobs = old_logprobs[mb_idx]
            mb_returns = returns[mb_idx]
            mb_adv = advantages[mb_idx]
            mb_raise_masks = raise_masks[mb_idx]

            logprob, entropy, value = policy.evaluate_actions(
                mb_obs,
                mb_masks,
                mb_actions,
                mb_bet_fracs,
                mb_raise_masks,
            )
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
    parser = argparse.ArgumentParser(description="PPO for custom NLHE (multi-player).")
    parser.add_argument("--num-players", type=int, default=2)
    parser.add_argument("--stack", type=int, default=20000)
    parser.add_argument("--small-blind", type=int, default=50)
    parser.add_argument("--big-blind", type=int, default=100)
    parser.add_argument("--max-raises", type=int, default=4)
    parser.add_argument("--ante", type=int, default=0)
    parser.add_argument("--rake-pct", type=float, default=0.0)
    parser.add_argument("--rake-cap", type=int, default=0)
    parser.add_argument("--rake-cap-hand", type=int, default=0)
    parser.add_argument("--rake-cap-street", type=int, default=0)
    parser.add_argument("--history-len", type=int, default=12)
    parser.add_argument("--hands-per-episode", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=200000)
    parser.add_argument("--rollout-episodes", type=int, default=200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=1024)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every", type=int, default=5000)
    parser.add_argument("--save-every", type=int, default=20000)
    parser.add_argument("--save-dir", default="experiments/custom_nlhe_ppo")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=1000)
    parser.add_argument("--eval-parallel", type=int, default=1)
    parser.add_argument("--eval-opponent", choices=["random", "lbr"], default="random")
    parser.add_argument("--lbr-rollouts", type=int, default=32)
    parser.add_argument("--lbr-bet-fracs", default="0.5,1.0,2.0")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_config = EnvConfig(
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
    )
    env = NoLimitHoldemEnv(env_config)
    obs_dim = env.obs_dim
    policy = TwoHeadPolicy(PolicyConfig(obs_dim=obs_dim, hidden_dim=args.hidden_dim), device=args.device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    total_episodes = 0
    if args.resume:
        state = torch.load(args.resume, map_location=args.device)
        policy.load_state_dict(state["model"])
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        total_episodes = int(state.get("episodes", 0))
        print(f"resume_from={args.resume} episodes={total_episodes}")

    start = time.time()
    print(
        "train_start "
        f"players={env_config.num_players} stack={env_config.stack} "
        f"blinds={env_config.small_blind}/{env_config.big_blind} "
        f"max_raises={env_config.max_raises_per_round} "
        f"ante={env_config.ante} rake={env_config.rake_pct} cap={env_config.rake_cap} "
        f"cap_hand={env_config.rake_cap_per_hand} cap_street={env_config.rake_cap_per_street} "
        f"history={env_config.history_len} hands_per_ep={env_config.hands_per_episode} "
        f"workers={args.workers} device={args.device}",
        flush=True,
    )
    while total_episodes < args.episodes:
        policy_state = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
        rollout_start = time.time()
        if args.workers > 1:
            per_worker = max(1, args.rollout_episodes // args.workers)
            payloads = [
                (policy_state, env_config, per_worker, args.seed + i, args.hidden_dim)
                for i in range(args.workers)
            ]
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=args.workers) as pool:
                buffers = pool.starmap(collect_rollouts, payloads)
            total_episodes += per_worker * args.workers
        else:
            buffers = [
                collect_rollouts(
                    policy_state,
                    env_config,
                    args.rollout_episodes,
                    args.seed,
                    args.hidden_dim,
                )
            ]
            total_episodes += args.rollout_episodes
        rollout_elapsed = time.time() - rollout_start
        merged = merge_buffers(buffers)
        batch = merged.as_tensors(policy.device)
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

        if args.log_every and total_episodes % args.log_every == 0:
            elapsed = time.time() - start
            print(
                f"episodes={total_episodes} elapsed={elapsed:.1f}s "
                f"rollout_sec={rollout_elapsed:.1f}",
                flush=True,
            )

        if args.save_every and total_episodes % args.save_every == 0:
            save_dir = pathlib.Path(args.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            path = save_dir / f"policy_ep_{total_episodes:06d}.pt"
            torch.save(
                {
                    "model": policy.state_dict(),
                    "episodes": total_episodes,
                    "optimizer": optimizer.state_dict(),
                    "obs_dim": obs_dim,
                    "config": env_config.__dict__,
                    "hidden_dim": args.hidden_dim,
                },
                path,
            )
            print(f"checkpoint_saved={path}")

        if args.eval_every and total_episodes % args.eval_every == 0:
            bet_fracs = [float(x) for x in args.lbr_bet_fracs.split(",") if x]
            score = evaler.evaluate_parallel(
                policy_state,
                env_config,
                args.eval_episodes,
                opponent_type=args.eval_opponent,
                eval_parallel=args.eval_parallel,
                lbr_rollouts=args.lbr_rollouts,
                lbr_bet_fracs=bet_fracs,
                hidden_dim=args.hidden_dim,
            )
            print(
                f"eval@{total_episodes} opponent={args.eval_opponent} avg_return={score:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
