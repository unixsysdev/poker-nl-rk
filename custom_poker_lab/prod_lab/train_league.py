from __future__ import annotations

import argparse
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np
import torch
from torch import nn, optim

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_poker_lab.poker_policy import PolicyConfig, TwoHeadPolicy, TrajectoryBuffer
from custom_poker_lab.prod_lab.vector_env import VectorEnvConfig, VectorNLHEEnv
from custom_poker_lab.prod_lab import eval as evaler


class RandomPolicy:
    def __init__(self, rng):
        self.rng = rng

    def act(self, obs, mask):
        actions = [i for i, v in enumerate(mask) if v > 0]
        if not actions:
            return 1, 0.0
        return int(self.rng.choice(actions)), float(self.rng.random())


def env_obs_dim(env_config: VectorEnvConfig) -> int:
    return 52 + 5 + 7 * env_config.num_players + 4 * env_config.history_len


def sample_actions(policy: TwoHeadPolicy, obs: torch.Tensor, mask: torch.Tensor):
    type_logits, size_params, value = policy.net(obs)
    type_logits = type_logits.masked_fill(mask == 0, -1e9)
    type_dist = torch.distributions.Categorical(logits=type_logits)
    action_type = type_dist.sample()
    logprob_type = type_dist.log_prob(action_type)
    entropy_type = type_dist.entropy()

    alpha = torch.nn.functional.softplus(size_params[:, 0]) + 1.0
    beta = torch.nn.functional.softplus(size_params[:, 1]) + 1.0
    size_dist = torch.distributions.Beta(alpha, beta)
    bet_frac = size_dist.sample().clamp(0.0, 1.0)
    logprob_size = size_dist.log_prob(bet_frac)
    entropy_size = size_dist.entropy()

    raise_mask = (action_type == 2).float()
    logprob = logprob_type + raise_mask * logprob_size
    entropy = entropy_type + raise_mask * entropy_size

    return action_type, bet_frac, logprob, value, entropy, raise_mask


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


def _split_batch_sizes(total: int, workers: int) -> list[int]:
    if workers <= 1:
        return [total]
    base = total // workers
    extra = total % workers
    sizes = [base + 1 if i < extra else base for i in range(workers)]
    return [size for size in sizes if size > 0]


def collect_rollouts_league(
    policy_state: dict,
    env_config_dict: dict,
    rollout_episodes: int,
    seed: int,
    hidden_dim: int,
    pool_states: list[dict],
    opponent_prob: float,
):
    env_config = VectorEnvConfig(**env_config_dict)
    env_config.seed = seed
    policy = TwoHeadPolicy(
        PolicyConfig(obs_dim=52 + 5 + 7 * env_config.num_players + 4 * env_config.history_len, hidden_dim=hidden_dim),
        device="cpu",
    )
    policy.load_state_dict(policy_state)

    opponent_policies = []
    for state in pool_states:
        opp = TwoHeadPolicy(
            PolicyConfig(obs_dim=52 + 5 + 7 * env_config.num_players + 4 * env_config.history_len, hidden_dim=hidden_dim),
            device="cpu",
        )
        opp.load_state_dict(state)
        opponent_policies.append(opp)

    rng = np.random.default_rng(seed)
    env = VectorNLHEEnv(env_config)
    return rollout_episode_steps(env, policy, opponent_policies, opponent_prob, rng, rollout_episodes)


def merge_buffers(buffers: list[TrajectoryBuffer]) -> TrajectoryBuffer:
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


def rollout_episode_steps(
    env: VectorNLHEEnv,
    policy: TwoHeadPolicy,
    opponent_policies: list[TwoHeadPolicy],
    opponent_prob: float,
    rng: np.random.Generator,
    episodes_per_env: int,
):
    obs_np, mask_np, current_players = env.get_obs()
    episode_steps = [[] for _ in range(env.config.batch_size)]
    step_player_ids = []
    returns = []
    batch = TrajectoryBuffer()

    episodes_done = np.zeros(env.config.batch_size, dtype=np.int64)
    while np.any(episodes_done < episodes_per_env):
        active_envs = np.where(~env.episode_over)[0]
        if active_envs.size == 0:
            for env_idx in range(env.config.batch_size):
                if episodes_done[env_idx] < episodes_per_env:
                    env.reset_at(env_idx)
            obs_np, mask_np, current_players = env.get_obs()
            continue
        obs_active = obs_np[active_envs]
        mask_active = mask_np[active_envs]
        players_active = current_players[active_envs]

        action_type_np = np.ones(env.config.batch_size, dtype=np.int64)
        bet_frac_np = np.zeros(env.config.batch_size, dtype=np.float32)

        learner_envs = [i for i, pid in zip(active_envs, players_active) if pid == 0]
        if learner_envs:
            idx = np.array(learner_envs, dtype=np.int64)
            obs = torch.tensor(obs_np[idx], dtype=torch.float32, device=policy.device)
            masks = torch.tensor(mask_np[idx], dtype=torch.float32, device=policy.device)
            action_type, bet_frac, logprob, value, entropy, raise_mask = sample_actions(policy, obs, masks)

            action_type_np[idx] = action_type.detach().cpu().numpy()
            bet_frac_np[idx] = bet_frac.detach().cpu().numpy()

            for env_idx in idx:
                step_player_ids.append(0)
                returns.append(None)
                episode_steps[env_idx].append(len(returns) - 1)

            batch.obs.extend(obs_np[idx])
            batch.masks.extend(mask_np[idx])
            batch.action_types.extend(action_type.detach().cpu().numpy().tolist())
            batch.bet_fracs.extend(bet_frac.detach().cpu().numpy().tolist())
            batch.logprobs.extend(logprob.detach().cpu())
            batch.values.extend(value.detach().cpu())
            batch.entropies.extend(entropy.detach().cpu())
            batch.raise_masks.extend(raise_mask.detach().cpu().tolist())
            batch.returns.extend([0.0] * len(idx))

        opponent_envs = [i for i, pid in zip(active_envs, players_active) if pid != 0]
        for env_idx in opponent_envs:
            obs = obs_np[env_idx]
            mask = mask_np[env_idx]
            use_pool = opponent_policies and rng.random() < opponent_prob
            if use_pool:
                opp = opponent_policies[int(rng.integers(0, len(opponent_policies)))]
                a_type, b_frac, *_ = opp.act({"obs": obs, "legal_action_mask": mask}, deterministic=True)
                action_type_np[env_idx] = a_type
                bet_frac_np[env_idx] = b_frac
            else:
                a_type, b_frac = RandomPolicy(rng).act(obs, mask)
                action_type_np[env_idx] = a_type
                bet_frac_np[env_idx] = b_frac

        obs_np, mask_np, current_players = env.step(action_type_np, bet_frac_np)

        for env_idx in range(env.config.batch_size):
            if env.episode_over[env_idx]:
                payoffs = env.get_payoffs()[env_idx]
                for idx in episode_steps[env_idx]:
                    returns[idx] = float(payoffs[0])
                episode_steps[env_idx] = []
                episodes_done[env_idx] += 1
                if episodes_done[env_idx] < episodes_per_env:
                    env.reset_at(env_idx)

    for i, ret in enumerate(returns):
        batch.returns[i] = 0.0 if ret is None else ret

    return batch


def train_candidate(
    base_state: dict,
    env_config: VectorEnvConfig,
    args,
    pool_states: list[dict],
    rollout_pool,
    batch_sizes: list[int],
):
    policy = TwoHeadPolicy(
        PolicyConfig(obs_dim=env_obs_dim(env_config), hidden_dim=args.hidden_dim),
        device=args.device,
    )
    policy.load_state_dict(base_state)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    opponent_policies = []
    for state in pool_states:
        opp = TwoHeadPolicy(PolicyConfig(obs_dim=env_obs_dim(env_config), hidden_dim=args.hidden_dim), device="cpu")
        opp.load_state_dict(state)
        opponent_policies.append(opp)

    env = VectorNLHEEnv(env_config)
    total_episodes = 0
    updates = 0
    while total_episodes < args.episodes_per_agent:
        rollout_start = time.time()
        if rollout_pool:
            policy_state = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
            payloads = []
            for i, batch_size in enumerate(batch_sizes):
                worker_config = env_config.__dict__.copy()
                worker_config["batch_size"] = batch_size
                worker_config["seed"] = args.seed + updates * args.rollout_workers + i
                worker_config["cpu_eval_workers"] = 0
                payloads.append(
                    (
                        policy_state,
                        worker_config,
                        args.rollout_episodes,
                        worker_config["seed"],
                        args.hidden_dim,
                        pool_states,
                        args.pool_prob,
                    )
                )
            buffers = rollout_pool.starmap(collect_rollouts_league, payloads)
            batch = merge_buffers(buffers)
            total_episodes += int(sum(batch_sizes) * args.rollout_episodes)
        else:
            batch = rollout_episode_steps(
                env,
                policy,
                opponent_policies,
                args.pool_prob,
                np.random.default_rng(args.seed),
                episodes_per_env=args.rollout_episodes,
            )
            total_episodes += int(env_config.batch_size * args.rollout_episodes)
        rollout_elapsed = time.time() - rollout_start
        if not batch.obs:
            env.reset()
            continue
        update_start = time.time()
        batch_tensors = batch.as_tensors(policy.device)
        ppo_update(
            policy,
            optimizer,
            batch_tensors,
            clip_ratio=args.clip_ratio,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            epochs=args.ppo_epochs,
            minibatch=args.minibatch,
        )
        update_elapsed = time.time() - update_start
        updates += 1
        if args.log_every and updates % args.log_every == 0:
            msg = (
                f"candidate_updates={updates} episodes={total_episodes} "
                f"rollout_sec={rollout_elapsed:.2f} ppo_sec={update_elapsed:.2f}"
            )
            if args.profile:
                samples = len(batch.obs)
                total_sec = max(1e-6, rollout_elapsed + update_elapsed)
                msg += f" samples={samples} samples_per_sec={samples/total_sec:.1f}"
            print(msg, flush=True)

    return policy.state_dict()


def main():
    parser = argparse.ArgumentParser(description="League training for prod NLHE (vectorized).")
    parser.add_argument("--batch-size", type=int, default=64)
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
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--episodes-per-agent", type=int, default=20000)
    parser.add_argument("--rollout-episodes", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=2048)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--rollout-workers", type=int, default=1)
    parser.add_argument("--cpu-eval-workers", type=int, default=0)
    parser.add_argument("--cpu-eval-min-batch", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=2000)
    parser.add_argument("--eval-opponent", choices=["random", "lbr", "dlbr", "proxy"], default="proxy")
    parser.add_argument("--lbr-rollouts", type=int, default=32)
    parser.add_argument("--lbr-bet-fracs", default="0.25,0.5,1.0")
    parser.add_argument("--pool-size", type=int, default=8)
    parser.add_argument("--pool-prob", type=float, default=0.5)
    parser.add_argument("--save-dir", default="experiments/prod_nlhe_league")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_config = VectorEnvConfig(
        batch_size=args.batch_size,
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
        cpu_eval_workers=args.cpu_eval_workers,
        cpu_eval_min_batch=args.cpu_eval_min_batch,
    )

    rollout_pool = None
    batch_sizes = _split_batch_sizes(args.batch_size, args.rollout_workers)
    if args.rollout_workers > 1:
        ctx = mp.get_context("spawn")
        rollout_pool = ctx.Pool(processes=args.rollout_workers)

    base_policy = TwoHeadPolicy(
        PolicyConfig(obs_dim=env_obs_dim(env_config), hidden_dim=args.hidden_dim),
        device=args.device,
    )
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        base_policy.load_state_dict(state["model"])
        print(f"resume_from={args.resume}")

    base_state = {k: v.detach().cpu() for k, v in base_policy.state_dict().items()}
    pool_states: list[dict] = []

    save_root = pathlib.Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    bet_fracs = [float(x) for x in args.lbr_bet_fracs.split(",") if x]

    for round_idx in range(args.rounds):
        round_start = time.time()
        print(f"round_start={round_idx} population={args.population}", flush=True)
        candidates = []
        scores = []
        for agent_idx in range(args.population):
            train_start = time.time()
            trained_state = train_candidate(base_state, env_config, args, pool_states, rollout_pool, batch_sizes)
            train_elapsed = time.time() - train_start
            candidates.append(trained_state)
            eval_start = time.time()
            score = evaler.evaluate(
                trained_state,
                env_config,
                args.eval_episodes,
                opponent=args.eval_opponent,
                lbr_rollouts=args.lbr_rollouts,
                lbr_bet_fracs=bet_fracs,
            )
            eval_elapsed = time.time() - eval_start
            scores.append(score)
            if args.profile:
                print(
                    f"round={round_idx} agent={agent_idx} score={score:.4f} "
                    f"train_sec={train_elapsed:.1f} eval_sec={eval_elapsed:.1f}",
                    flush=True,
                )
            else:
                print(f"round={round_idx} agent={agent_idx} score={score:.4f}", flush=True)

            agent_dir = save_root / f"round_{round_idx:02d}"
            agent_dir.mkdir(parents=True, exist_ok=True)
            path = agent_dir / f"agent_{agent_idx:02d}.pt"
            torch.save(
                {
                    "model": trained_state,
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
        if args.profile:
            round_elapsed = time.time() - round_start
            print(
                f"round_end={round_idx} best_score={best_score:.4f} "
                f"round_sec={round_elapsed:.1f}",
                flush=True,
            )
        else:
            print(f"round_end={round_idx} best_score={best_score:.4f}", flush=True)

    if rollout_pool:
        rollout_pool.close()
        rollout_pool.join()


if __name__ == "__main__":
    main()
