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
from custom_poker_lab.prod_lab.vector_env import VectorEnvConfig, VectorNLHEEnv
from custom_poker_lab.prod_lab import eval as evaler


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


def main():
    parser = argparse.ArgumentParser(description="Vectorized PPO for production-grade NLHE env.")
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
    parser.add_argument("--episodes", type=int, default=200000)
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
    parser.add_argument("--log-every", type=int, default=10000)
    parser.add_argument("--log-every-updates", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=50000)
    parser.add_argument("--save-dir", default="experiments/prod_nlhe_ppo")
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=2000)
    parser.add_argument("--eval-opponent", choices=["random", "lbr", "proxy"], default="random")
    parser.add_argument("--lbr-rollouts", type=int, default=32)
    parser.add_argument("--lbr-bet-fracs", default="0.25,0.5,1.0")
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
    )
    env = VectorNLHEEnv(env_config)
    policy = TwoHeadPolicy(
        PolicyConfig(obs_dim=env.obs_dim, hidden_dim=args.hidden_dim),
        device=args.device,
    )
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    total_episodes = 0
    start = time.time()
    print(
        "train_start "
        f"batch={env_config.batch_size} players={env_config.num_players} "
        f"stack={env_config.stack} blinds={env_config.small_blind}/{env_config.big_blind} "
        f"max_raises={env_config.max_raises_per_round} "
        f"hands_per_ep={env_config.hands_per_episode} "
        f"device={args.device}",
        flush=True,
    )

    obs_np, mask_np, current_players = env.get_obs()
    episode_steps = [[] for _ in range(env_config.batch_size)]
    step_env_ids = []
    step_player_ids = []
    returns = []

    update_idx = 0
    while total_episodes < args.episodes:
        batch = TrajectoryBuffer()
        episode_steps = [[] for _ in range(env_config.batch_size)]
        step_env_ids = []
        step_player_ids = []
        returns = []
        episodes_done = np.zeros(env_config.batch_size, dtype=np.int64)

        rollout_start = time.time()
        while np.any(episodes_done < args.rollout_episodes):
            active_envs = np.where(~env.episode_over)[0]
            if active_envs.size == 0:
                for env_idx in range(env_config.batch_size):
                    if episodes_done[env_idx] < args.rollout_episodes:
                        env.reset_at(env_idx)
                obs_np, mask_np, current_players = env.get_obs()
                continue
            obs_active = obs_np[active_envs]
            mask_active = mask_np[active_envs]

            obs = torch.tensor(obs_active, dtype=torch.float32, device=policy.device)
            masks = torch.tensor(mask_active, dtype=torch.float32, device=policy.device)
            action_type, bet_frac, logprob, value, entropy, raise_mask = sample_actions(policy, obs, masks)

            action_type_np = np.ones(env_config.batch_size, dtype=np.int64)
            bet_frac_np = np.zeros(env_config.batch_size, dtype=np.float32)
            action_type_np[active_envs] = action_type.detach().cpu().numpy()
            bet_frac_np[active_envs] = bet_frac.detach().cpu().numpy()

            for idx, env_idx in enumerate(active_envs):
                step_env_ids.append(int(env_idx))
                step_player_ids.append(int(current_players[env_idx]))
                returns.append(None)
                episode_steps[env_idx].append(len(returns) - 1)

            batch.obs.extend(obs_active)
            batch.masks.extend(mask_active)
            batch.action_types.extend(action_type.detach().cpu().numpy().tolist())
            batch.bet_fracs.extend(bet_frac.detach().cpu().numpy().tolist())
            batch.logprobs.extend(logprob.detach().cpu())
            batch.values.extend(value.detach().cpu())
            batch.entropies.extend(entropy.detach().cpu())
            batch.raise_masks.extend(raise_mask.detach().cpu().tolist())
            batch.returns.extend([0.0] * obs_active.shape[0])

            obs_np, mask_np, current_players = env.step(action_type_np, bet_frac_np)

            for env_idx in range(env_config.batch_size):
                if env.episode_over[env_idx]:
                    payoffs = env.get_payoffs()[env_idx]
                    for idx in episode_steps[env_idx]:
                        player = step_player_ids[idx]
                        returns[idx] = float(payoffs[player])
                    episode_steps[env_idx] = []
                    episodes_done[env_idx] += 1
                    if episodes_done[env_idx] < args.rollout_episodes:
                        env.reset_at(env_idx)

        for i, ret in enumerate(returns):
            if ret is None:
                ret = 0.0
            batch.returns[i] = ret

        if not batch.obs:
            obs_np, mask_np, current_players = env.get_obs()
            continue

        total_episodes += int(env_config.batch_size * args.rollout_episodes)
        batch_tensors = batch.as_tensors(policy.device)
        update_start = time.time()
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
        update_idx += 1
        rollout_elapsed = update_start - rollout_start
        update_elapsed = time.time() - update_start

        if args.log_every and total_episodes % args.log_every == 0:
            elapsed = time.time() - start
            print(
                f"episodes={total_episodes} updates={update_idx} elapsed={elapsed:.1f}s "
                f"rollout_sec={rollout_elapsed:.2f} ppo_sec={update_elapsed:.2f}",
                flush=True,
            )
        if args.log_every_updates and update_idx % args.log_every_updates == 0:
            elapsed = time.time() - start
            print(
                f"update={update_idx} episodes={total_episodes} elapsed={elapsed:.1f}s "
                f"rollout_sec={rollout_elapsed:.2f} ppo_sec={update_elapsed:.2f}",
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
                    "obs_dim": env.obs_dim,
                    "config": env_config.__dict__,
                    "hidden_dim": args.hidden_dim,
                },
                path,
            )
            print(f"checkpoint_saved={path}")

        if args.eval_every and total_episodes % args.eval_every == 0:
            bet_fracs = [float(x) for x in args.lbr_bet_fracs.split(",") if x]
            score = evaler.evaluate(
                policy.state_dict(),
                env_config,
                args.eval_episodes,
                opponent=args.eval_opponent,
                lbr_rollouts=args.lbr_rollouts,
                lbr_bet_fracs=bet_fracs,
            )
            print(f"eval@{total_episodes} opponent={args.eval_opponent} avg_return={score:.4f}", flush=True)


if __name__ == "__main__":
    main()
