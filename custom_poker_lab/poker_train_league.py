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


class RandomPolicy:
    def __init__(self, rng):
        self.rng = rng

    def act(self, state):
        legal = state["legal_action_mask"]
        actions = [i for i, v in enumerate(legal) if v > 0]
        action_type = int(self.rng.choice(actions))
        bet_frac = float(self.rng.random())
        return action_type, bet_frac


def run_episode(env: NoLimitHoldemEnv, policies: list, learn_players: list[int]):
    buffers = {pid: TrajectoryBuffer() for pid in learn_players}
    state, player_id = env.reset()
    while not env.is_over():
        policy = policies[player_id]
        if player_id in learn_players:
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
        else:
            action_type, bet_frac = policy.act(state)
        state, player_id = env.step(action_type, bet_frac)
    payoffs = env.get_payoffs()
    for pid in learn_players:
        buffers[pid].returns = [float(payoffs[pid])] * len(buffers[pid].returns)
    return list(buffers.values())


def collect_rollouts(
    payload,
):
    (
        policy_state,
        env_config,
        episodes,
        seed,
        hidden_dim,
        pool_states,
        pool_prob,
    ) = payload
    rng = np.random.default_rng(seed)
    policy = TwoHeadPolicy(
        PolicyConfig(obs_dim=env_config_obs_dim(env_config), hidden_dim=hidden_dim),
        device="cpu",
    )
    policy.load_state_dict(policy_state)
    local_config = replace(env_config, seed=seed)
    env = NoLimitHoldemEnv(local_config)

    pool_policies = []
    for state in pool_states:
        opp = TwoHeadPolicy(
            PolicyConfig(obs_dim=env.obs_dim, hidden_dim=hidden_dim),
            device="cpu",
        )
        opp.load_state_dict(state)
        pool_policies.append(opp)

    merged = TrajectoryBuffer()
    for _ in range(episodes):
        policies = []
        for pid in range(env.config.num_players):
            if pid == 0:
                policies.append(policy)
                continue
            use_pool = pool_policies and rng.random() < pool_prob
            if use_pool:
                policies.append(pool_policies[int(rng.integers(0, len(pool_policies)))])
            else:
                policies.append(RandomPolicy(rng))
        buffers = run_episode(env, policies, learn_players=[0])
        buf = buffers[0]
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


def train_candidate(
    base_state,
    env_config,
    args,
    pool_states,
):
    policy = TwoHeadPolicy(
        PolicyConfig(obs_dim=env_config_obs_dim(env_config), hidden_dim=args.hidden_dim),
        device=args.device,
    )
    policy.load_state_dict(base_state)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    total_episodes = 0
    while total_episodes < args.episodes_per_agent:
        policy_state = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
        rollout_start = time.time()
        if args.workers > 1:
            per_worker = max(1, args.rollout_episodes // args.workers)
            payloads = [
                (
                    policy_state,
                    env_config,
                    per_worker,
                    args.seed + i,
                    args.hidden_dim,
                    pool_states,
                    args.pool_prob,
                )
                for i in range(args.workers)
            ]
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=args.workers) as pool:
                buffers = pool.map(collect_rollouts, payloads)
            total_episodes += per_worker * args.workers
        else:
            buffers = [
                collect_rollouts(
                    policy_state,
                    env_config,
                    args.rollout_episodes,
                    args.seed,
                    args.hidden_dim,
                    pool_states,
                    args.pool_prob,
                )
            ]
            total_episodes += args.rollout_episodes

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
            rollout_elapsed = time.time() - rollout_start
            print(
                f"train_agent episodes={total_episodes} rollout_sec={rollout_elapsed:.1f}",
                flush=True,
            )

    return policy.state_dict()


def main():
    parser = argparse.ArgumentParser(description="League training for custom NLHE.")
    parser.add_argument("--num-players", type=int, default=4)
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
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--episodes-per-agent", type=int, default=20000)
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
    parser.add_argument("--eval-episodes", type=int, default=2000)
    parser.add_argument("--eval-parallel", type=int, default=1)
    parser.add_argument("--eval-opponent", choices=["random", "lbr"], default="random")
    parser.add_argument("--lbr-rollouts", type=int, default=32)
    parser.add_argument("--lbr-bet-fracs", default="0.5,1.0,2.0")
    parser.add_argument("--pool-size", type=int, default=8)
    parser.add_argument("--pool-prob", type=float, default=0.5)
    parser.add_argument("--save-dir", default="experiments/custom_nlhe_league")
    parser.add_argument("--resume", default=None)
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

    base_policy = TwoHeadPolicy(
        PolicyConfig(obs_dim=env_config_obs_dim(env_config), hidden_dim=args.hidden_dim),
        device="cpu",
    )
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        base_policy.load_state_dict(state["model"])
        print(f"resume_from={args.resume}")

    base_state = {k: v.detach().cpu() for k, v in base_policy.state_dict().items()}
    pool_states: list[dict] = []

    save_root = pathlib.Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    for round_idx in range(args.rounds):
        print(f"round_start={round_idx} population={args.population}", flush=True)
        candidates = []
        for agent_idx in range(args.population):
            trained_state = train_candidate(base_state, env_config, args, pool_states)
            candidates.append(trained_state)

        bet_fracs = [float(x) for x in args.lbr_bet_fracs.split(",") if x]
        scores = []
        for agent_idx, state in enumerate(candidates):
            score = evaler.evaluate_parallel(
                state,
                env_config,
                args.eval_episodes,
                opponent_type=args.eval_opponent,
                eval_parallel=args.eval_parallel,
                lbr_rollouts=args.lbr_rollouts,
                lbr_bet_fracs=bet_fracs,
                hidden_dim=args.hidden_dim,
            )
            scores.append(score)
            print(f"round={round_idx} agent={agent_idx} score={score:.4f}", flush=True)

            agent_dir = save_root / f"round_{round_idx:02d}"
            agent_dir.mkdir(parents=True, exist_ok=True)
            path = agent_dir / f"agent_{agent_idx:02d}.pt"
            torch.save(
                {
                    "model": state,
                    "config": env_config.__dict__,
                    "hidden_dim": args.hidden_dim,
                },
                path,
            )

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top = ranked[: max(1, args.top_k)]
        base_state = candidates[top[0]]

        for idx in top:
            pool_states.append(candidates[idx])
        if args.pool_size and len(pool_states) > args.pool_size:
            pool_states = pool_states[-args.pool_size :]

        best_score = scores[top[0]]
        print(f"round_end={round_idx} best_score={best_score:.4f}", flush=True)


if __name__ == "__main__":
    main()
