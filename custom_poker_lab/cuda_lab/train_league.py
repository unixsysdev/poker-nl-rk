from __future__ import annotations

import argparse
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
from custom_poker_lab.cuda_lab.cuda_env import CudaEnvConfig, CudaNLHEEnv
from custom_poker_lab.cuda_lab import eval as evaler


class RandomPolicy:
    def __init__(self, rng):
        self.rng = rng

    def act(self, mask):
        actions = [i for i, v in enumerate(mask) if v > 0]
        if not actions:
            return 1, 0.0
        return int(self.rng.choice(actions)), float(self.rng.random())


def env_obs_dim(env_config: CudaEnvConfig) -> int:
    return 52 + 5 + 7 * env_config.num_players + 4 * env_config.history_len


def sample_actions(policy: TwoHeadPolicy, obs: torch.Tensor, mask: torch.Tensor, deterministic: bool = False):
    type_logits, size_params, value = policy.net(obs)
    type_logits = type_logits.masked_fill(mask == 0, -1e9)
    type_dist = torch.distributions.Categorical(logits=type_logits)
    if deterministic:
        action_type = torch.argmax(type_dist.probs, dim=-1)
    else:
        action_type = type_dist.sample()
    logprob_type = type_dist.log_prob(action_type)
    entropy_type = type_dist.entropy()

    alpha = torch.nn.functional.softplus(size_params[:, 0]) + 1.0
    beta = torch.nn.functional.softplus(size_params[:, 1]) + 1.0
    size_dist = torch.distributions.Beta(alpha, beta)
    if deterministic:
        bet_frac = (alpha / (alpha + beta)).clamp(0.0, 1.0)
    else:
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


def rollout_episode_steps(
    env: CudaNLHEEnv,
    policy: TwoHeadPolicy,
    opponent_policies: list[TwoHeadPolicy],
    opponent_prob: float,
    rng: np.random.Generator,
    episodes_per_env: int,
):
    obs, mask, current = env.get_obs()
    episode_steps = [[] for _ in range(env.config.batch_size)]
    step_player_ids = []
    returns = []
    batch = TrajectoryBuffer()
    episodes_done = torch.zeros(env.config.batch_size, device=env.device, dtype=torch.int64)

    while torch.any(episodes_done < episodes_per_env):
        active_envs = torch.where(~env.episode_over)[0]
        if active_envs.numel() == 0:
            for env_idx in range(env.config.batch_size):
                if int(episodes_done[env_idx].item()) < episodes_per_env:
                    env.reset_at(env_idx)
            obs, mask, current = env.get_obs()
            continue

        action_types = torch.ones(env.config.batch_size, device=env.device, dtype=torch.int64)
        bet_fracs = torch.zeros(env.config.batch_size, device=env.device, dtype=torch.float32)

        learner_envs = [i for i in active_envs.tolist() if int(current[i].item()) == 0]
        if learner_envs:
            idx = torch.tensor(learner_envs, device=env.device, dtype=torch.int64)
            obs_active = obs[idx]
            mask_active = mask[idx]
            action_type, bet_frac, logprob, value, entropy, raise_mask = sample_actions(
                policy, obs_active, mask_active
            )
            action_types[idx] = action_type
            bet_fracs[idx] = bet_frac

            batch.obs.extend(obs_active.detach().cpu().numpy())
            batch.masks.extend(mask_active.detach().cpu().numpy())
            batch.action_types.extend(action_type.detach().cpu().numpy().tolist())
            batch.bet_fracs.extend(bet_frac.detach().cpu().numpy().tolist())
            batch.logprobs.extend(logprob.detach().cpu())
            batch.values.extend(value.detach().cpu())
            batch.entropies.extend(entropy.detach().cpu())
            batch.raise_masks.extend(raise_mask.detach().cpu().tolist())
            batch.returns.extend([0.0] * obs_active.shape[0])

            for env_idx in learner_envs:
                step_player_ids.append(0)
                returns.append(None)
                episode_steps[env_idx].append(len(returns) - 1)

        opponent_envs = [i for i in active_envs.tolist() if int(current[i].item()) != 0]
        for env_idx in opponent_envs:
            use_pool = opponent_policies and rng.random() < opponent_prob
            if use_pool:
                opp = opponent_policies[int(rng.integers(0, len(opponent_policies)))]
                a_type, b_frac, *_ = sample_actions(
                    opp,
                    obs[env_idx : env_idx + 1],
                    mask[env_idx : env_idx + 1],
                    deterministic=True,
                )
                action_types[env_idx] = a_type[0]
                bet_fracs[env_idx] = b_frac[0]
            else:
                a_type, b_frac = RandomPolicy(rng).act(mask[env_idx].detach().cpu().numpy())
                action_types[env_idx] = a_type
                bet_fracs[env_idx] = b_frac

        obs, mask, current = env.step(action_types, bet_fracs)

        for env_idx in range(env.config.batch_size):
            if bool(env.episode_over[env_idx].item()):
                payoffs = env.get_payoffs()[env_idx]
                for idx in episode_steps[env_idx]:
                    returns[idx] = float(payoffs[0].item())
                episode_steps[env_idx] = []
                episodes_done[env_idx] += 1
                if int(episodes_done[env_idx].item()) < episodes_per_env:
                    env.reset_at(env_idx)

    for i, ret in enumerate(returns):
        batch.returns[i] = 0.0 if ret is None else ret

    return batch


def train_candidate(base_state: dict, env_config: CudaEnvConfig, args, pool_states: list[dict]):
    policy = TwoHeadPolicy(
        PolicyConfig(obs_dim=env_obs_dim(env_config), hidden_dim=args.hidden_dim),
        device=args.device,
    )
    policy.load_state_dict(base_state)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    opponent_policies = []
    for state in pool_states:
        opp = TwoHeadPolicy(PolicyConfig(obs_dim=env_obs_dim(env_config), hidden_dim=args.hidden_dim), device=args.device)
        opp.load_state_dict(state)
        opponent_policies.append(opp)

    env = CudaNLHEEnv(env_config)
    total_episodes = 0
    updates = 0
    while total_episodes < args.episodes_per_agent:
        batch = rollout_episode_steps(
            env,
            policy,
            opponent_policies,
            args.pool_prob,
            np.random.default_rng(args.seed),
            episodes_per_env=args.rollout_episodes,
        )
        if not batch.obs:
            env.reset()
            continue
        total_episodes += int(env_config.batch_size * args.rollout_episodes)
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
        updates += 1
        if args.log_every and updates % args.log_every == 0:
            print(f"candidate_updates={updates} episodes={total_episodes}", flush=True)

    return policy.state_dict()


def main():
    parser = argparse.ArgumentParser(description="CUDA league training for NLHE.")
    parser.add_argument("--batch-size", type=int, default=128)
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
    parser.add_argument("--minibatch", type=int, default=4096)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=2000)
    parser.add_argument("--eval-opponent", choices=["random", "lbr", "dlbr", "proxy"], default="proxy")
    parser.add_argument("--lbr-rollouts", type=int, default=32)
    parser.add_argument("--lbr-bet-fracs", default="0.25,0.5,1.0")
    parser.add_argument("--br-depth", type=int, default=2)
    parser.add_argument("--br-other-samples", type=int, default=1)
    parser.add_argument("--pool-size", type=int, default=8)
    parser.add_argument("--pool-prob", type=float, default=0.5)
    parser.add_argument("--save-dir", default="experiments/cuda_nlhe_league")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    env_config = CudaEnvConfig(
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
        device=args.device,
    )

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
        print(f"round_start={round_idx} population={args.population}", flush=True)
        candidates = []
        scores = []
        for agent_idx in range(args.population):
            trained_state = train_candidate(base_state, env_config, args, pool_states)
            candidates.append(trained_state)
            score = evaler.evaluate(
                trained_state,
                env_config,
                args.eval_episodes,
                opponent=args.eval_opponent,
                lbr_rollouts=args.lbr_rollouts,
                lbr_bet_fracs=bet_fracs,
                br_depth=args.br_depth,
                br_other_samples=args.br_other_samples,
            )
            scores.append(score)
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
        print(f"round_end={round_idx} best_score={best_score:.4f}", flush=True)


if __name__ == "__main__":
    main()
