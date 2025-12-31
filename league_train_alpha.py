import argparse
import multiprocessing as mp
import os
import pathlib
import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

import pyspiel

from poker_rl.alpha_policy import PolicyNet
from poker_rl.solvers import CFRPolicy


@dataclass
class Trajectory:
    logprobs: List[torch.Tensor]
    values: List[torch.Tensor]


def _sample_action(policy, obs, legal_actions, device, rng):
    obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
    logits, value = policy(obs_t)
    mask = torch.full_like(logits, -1e9)
    mask[:, legal_actions] = 0.0
    logits = logits + mask
    dist = Categorical(logits=logits)
    action = dist.sample()
    logprob = dist.log_prob(action)
    return int(action.item()), logprob.squeeze(0), value.squeeze(0)


def _play_episode(game, policy_a, policy_b, device, rng, seat_a: int):
    state = game.new_initial_state()
    trajectories = {0: Trajectory([], []), 1: Trajectory([], [])}

    while not state.is_terminal():
        if state.is_chance_node():
            actions, probs = zip(*state.chance_outcomes())
            action = int(rng.choice(actions, p=probs))
            state.apply_action(action)
            continue

        player = state.current_player()
        legal_actions = state.legal_actions(player)
        obs = np.asarray(state.information_state_tensor(player), dtype=np.float32)

        policy = policy_a if player == seat_a else policy_b
        if isinstance(policy, CFRPolicy):
            action = policy.action(state)
        else:
            action, logprob, value = _sample_action(policy, obs, legal_actions, device, rng)
            trajectories[player].logprobs.append(logprob)
            trajectories[player].values.append(value)
        state.apply_action(action)

    returns = state.returns()
    return trajectories, returns


def _policy_loss(traj: Trajectory, target_return: float, value_coef: float):
    if not traj.logprobs:
        return torch.tensor(0.0)
    values = torch.stack(traj.values)
    logprobs = torch.stack(traj.logprobs)
    returns = torch.full_like(values, float(target_return))
    advantages = returns - values
    policy_loss = -(logprobs * advantages.detach()).mean()
    value_loss = 0.5 * advantages.pow(2).mean()
    return policy_loss + value_coef * value_loss


def train_policy(
    game,
    policy,
    opponents,
    optimizer,
    device,
    rng,
    episodes,
    value_coef,
    log_every,
    label,
    batch_episodes,
):
    policy.train()
    running = []
    batch_losses = []
    for _ in range(episodes):
        opponent = rng.choice(opponents)
        seat_a = int(rng.integers(0, 2))
        trajectories, returns = _play_episode(game, policy, opponent, device, rng, seat_a)
        traj = trajectories[seat_a]
        loss = _policy_loss(traj, returns[seat_a], value_coef)
        batch_losses.append(loss)
        if len(batch_losses) >= batch_episodes:
            optimizer.zero_grad()
            torch.stack(batch_losses).mean().backward()
            optimizer.step()
            batch_losses.clear()
        if log_every:
            running.append(returns[seat_a])
            if len(running) >= log_every:
                avg_return = float(np.mean(running))
                print(f"{label} avg_return={avg_return:.4f}")
                running.clear()
    if batch_losses:
        optimizer.zero_grad()
        torch.stack(batch_losses).mean().backward()
        optimizer.step()


def evaluate_head_to_head(game, policy_a, policy_b, device, rng, episodes):
    if isinstance(policy_a, nn.Module):
        policy_a.eval()
    if isinstance(policy_b, nn.Module):
        policy_b.eval()
    scores = []
    for i in range(episodes):
        seat_a = i % 2
        trajectories, returns = _play_episode(game, policy_a, policy_b, device, rng, seat_a)
        scores.append(returns[seat_a])
    return float(np.mean(scores))


def round_robin_scores(game, policies, device, rng, episodes):
    scores = np.zeros(len(policies), dtype=np.float32)
    for i in range(len(policies)):
        for j in range(i + 1, len(policies)):
            score = evaluate_head_to_head(game, policies[i], policies[j], device, rng, episodes)
            scores[i] += score
            scores[j] -= score
    return scores


def cfr_scores(game_name, policies, device, rng, episodes, cfr_iterations, cfr_algorithm):
    cfr_policy = CFRPolicy(game_name, iterations=cfr_iterations, algorithm=cfr_algorithm)
    game = pyspiel.load_game(game_name)
    scores = np.zeros(len(policies), dtype=np.float32)
    for i, policy in enumerate(policies):
        score = evaluate_head_to_head(game, policy, cfr_policy, device, rng, episodes)
        scores[i] = score
    return scores


def _eval_match_worker(payload):
    (
        game_name,
        policies,
        policy_pairs,
        episodes,
        seed,
        hidden_size,
    ) = payload

    rng = np.random.default_rng(seed)
    game = pyspiel.load_game(game_name)
    obs_dim = game.information_state_tensor_size()
    action_dim = game.num_distinct_actions()

    loaded = []
    for state in policies:
        policy = PolicyNet(obs_dim, action_dim, hidden_size)
        policy.load_state_dict(state)
        loaded.append(policy)

    scores = np.zeros(len(loaded), dtype=np.float32)
    for i, j in policy_pairs:
        score = evaluate_head_to_head(game, loaded[i], loaded[j], "cpu", rng, episodes)
        scores[i] += score
        scores[j] -= score
    return scores


def _eval_vs_cfr_worker(payload):
    (
        game_name,
        policy_state,
        episodes,
        seed,
        hidden_size,
        cfr_iterations,
        cfr_algorithm,
    ) = payload

    rng = np.random.default_rng(seed)
    game = pyspiel.load_game(game_name)
    obs_dim = game.information_state_tensor_size()
    action_dim = game.num_distinct_actions()

    policy = PolicyNet(obs_dim, action_dim, hidden_size)
    policy.load_state_dict(policy_state)
    cfr_policy = CFRPolicy(game_name, iterations=cfr_iterations, algorithm=cfr_algorithm)
    score = evaluate_head_to_head(game, policy, cfr_policy, "cpu", rng, episodes)
    return float(score)


def clone_policy(policy):
    clone = PolicyNet(
        obs_dim=policy.net[0].in_features,
        action_dim=policy.policy_head.out_features,
        hidden_size=policy.net[0].out_features,
    )
    state_dict = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
    clone.load_state_dict(state_dict)
    return clone


def _train_candidate_worker(payload):
    (
        game_name,
        obs_dim,
        action_dim,
        hidden_size,
        base_state,
        hof_states,
        episodes,
        value_coef,
        lr,
        seed,
        device,
        batch_episodes,
        log_every,
        label,
        threads,
    ) = payload

    if threads is not None:
        torch.set_num_threads(threads)
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)

    rng = np.random.default_rng(seed)
    game = pyspiel.load_game(game_name)
    policy = PolicyNet(obs_dim, action_dim, hidden_size).to(device)
    policy.load_state_dict(base_state)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    opponents = []
    for state in hof_states:
        opponent = PolicyNet(obs_dim, action_dim, hidden_size).to(device)
        opponent.load_state_dict(state)
        opponents.append(opponent)

    train_policy(
        game,
        policy,
        opponents,
        optimizer,
        device,
        rng,
        episodes,
        value_coef,
        log_every,
        label,
        batch_episodes,
    )
    final_state = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
    return final_state


def main():
    parser = argparse.ArgumentParser(description="Alpha-style population training (no PufferLib).")
    parser.add_argument("--game", default="leduc_poker")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--episodes-per-agent", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--worker-device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection-metric", choices=["round_robin", "cfr", "combined"], default="combined")
    parser.add_argument("--cfr-iterations", type=int, default=1000)
    parser.add_argument("--cfr-algorithm", choices=["cfr", "cfr_plus"], default="cfr")
    parser.add_argument("--cfr-weight", type=float, default=1.0)
    parser.add_argument("--output-dir", default="experiments/alpha_league")
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--batch-episodes", type=int, default=1)
    parser.add_argument("--parallel-agents", type=int, default=1)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--resume-model", default=None)
    parser.add_argument("--auto-resume", dest="auto_resume", action="store_true", default=True)
    parser.add_argument("--no-auto-resume", dest="auto_resume", action="store_false")
    parser.add_argument("--eval-parallel", type=int, default=1)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    game = pyspiel.load_game(args.game)

    obs_dim = game.information_state_tensor_size()
    action_dim = game.num_distinct_actions()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def find_latest_checkpoint(path: pathlib.Path):
        pattern = re.compile(r"round_(\d+)\.pt$")
        latest_round = None
        latest_path = None
        for checkpoint in path.glob("round_*.pt"):
            match = pattern.search(checkpoint.name)
            if not match:
                continue
            round_idx = int(match.group(1))
            if latest_round is None or round_idx > latest_round:
                latest_round = round_idx
                latest_path = checkpoint
        return latest_round, latest_path

    def round_index_from_path(path: str | None):
        if not path:
            return None
        match = re.search(r"round_(\d+)\.pt$", os.path.basename(path))
        return int(match.group(1)) if match else None

    start_round = 0
    resume_path = args.resume_model
    if resume_path is None and args.auto_resume:
        latest_round, latest_path = find_latest_checkpoint(output_dir)
        if latest_path is not None:
            resume_path = str(latest_path)
            start_round = latest_round + 1

    base_policy = PolicyNet(obs_dim, action_dim, args.hidden_size).to(args.device)
    if resume_path:
        state = torch.load(resume_path, map_location=args.device)
        base_policy.load_state_dict(state)
        if start_round == 0:
            parsed_round = round_index_from_path(resume_path)
            if parsed_round is not None:
                start_round = parsed_round + 1

    hall_of_fame = [clone_policy(base_policy).to(args.device)]

    worker_device = args.worker_device or ("cpu" if args.parallel_agents > 1 else args.device)

    for round_idx in range(start_round, start_round + args.rounds):
        candidates = []
        base_state = {k: v.detach().cpu() for k, v in base_policy.state_dict().items()}
        hof_states = [
            {k: v.detach().cpu() for k, v in policy.state_dict().items()}
            for policy in hall_of_fame
        ]

        if args.parallel_agents > 1:
            ctx = mp.get_context("spawn")
            payloads = []
            for agent_idx in range(args.population):
                payloads.append(
                    (
                        args.game,
                        obs_dim,
                        action_dim,
                        args.hidden_size,
                        base_state,
                        hof_states,
                        args.episodes_per_agent,
                        args.value_coef,
                        args.lr,
                        args.seed + round_idx * 10_000 + agent_idx,
                        worker_device,
                        args.batch_episodes,
                        args.log_every,
                        f"round={round_idx} agent={agent_idx}",
                        args.threads_per_worker,
                    )
                )
            with ctx.Pool(processes=args.parallel_agents) as pool:
                results = pool.map(_train_candidate_worker, payloads)

            for state in results:
                policy = PolicyNet(obs_dim, action_dim, args.hidden_size).to(args.device)
                policy.load_state_dict(state)
                candidates.append(policy)
        else:
            for agent_idx in range(args.population):
                policy = clone_policy(base_policy).to(args.device)
                optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
                train_policy(
                    game,
                    policy,
                    hall_of_fame,
                    optimizer,
                    args.device,
                    rng,
                    args.episodes_per_agent,
                    args.value_coef,
                    args.log_every,
                    f"round={round_idx} agent={agent_idx}",
                    args.batch_episodes,
                )
                candidates.append(policy)

        rr_scores = np.zeros(len(candidates), dtype=np.float32)
        if args.eval_parallel > 1:
            policy_states = [
                {k: v.detach().cpu() for k, v in policy.state_dict().items()}
                for policy in candidates
            ]
            pairs = [(i, j) for i in range(len(candidates)) for j in range(i + 1, len(candidates))]
            if pairs:
                chunks = [[] for _ in range(args.eval_parallel)]
                for idx, pair in enumerate(pairs):
                    chunks[idx % args.eval_parallel].append(pair)
                ctx = mp.get_context("spawn")
                payloads = []
                for idx, chunk in enumerate(chunks):
                    if not chunk:
                        continue
                    payloads.append(
                        (
                            args.game,
                            policy_states,
                            chunk,
                            args.eval_episodes,
                            args.seed + 50_000 + round_idx * 100 + idx,
                            args.hidden_size,
                        )
                    )
                with ctx.Pool(processes=args.eval_parallel) as pool:
                    results = pool.map(_eval_match_worker, payloads)
                for partial in results:
                    rr_scores += partial
        else:
            rr_scores = round_robin_scores(game, candidates, args.device, rng, args.eval_episodes)
        cfr_eval_scores = np.zeros_like(rr_scores)
        if args.selection_metric in {"cfr", "combined"}:
            if args.eval_parallel > 1:
                policy_states = [
                    {k: v.detach().cpu() for k, v in policy.state_dict().items()}
                    for policy in candidates
                ]
                ctx = mp.get_context("spawn")
                payloads = []
                for idx, state in enumerate(policy_states):
                    payloads.append(
                        (
                            args.game,
                            state,
                            args.eval_episodes,
                            args.seed + 60_000 + round_idx * 100 + idx,
                            args.hidden_size,
                            args.cfr_iterations,
                            args.cfr_algorithm,
                        )
                    )
                with ctx.Pool(processes=args.eval_parallel) as pool:
                    results = pool.map(_eval_vs_cfr_worker, payloads)
                cfr_eval_scores = np.asarray(results, dtype=np.float32)
            else:
                cfr_eval_scores = cfr_scores(
                    args.game,
                    candidates,
                    args.device,
                    rng,
                    args.eval_episodes,
                    args.cfr_iterations,
                    args.cfr_algorithm,
                )

        if args.selection_metric == "round_robin":
            scores = rr_scores
        elif args.selection_metric == "cfr":
            scores = cfr_eval_scores
        else:
            scores = rr_scores + args.cfr_weight * cfr_eval_scores

        top_indices = np.argsort(scores)[::-1][: args.top_k]
        base_policy = clone_policy(candidates[top_indices[0]]).to(args.device)

        hall_of_fame = [clone_policy(candidates[i]).to(args.device) for i in top_indices]

        round_path = output_dir / f"round_{round_idx:02d}.pt"
        torch.save(base_policy.state_dict(), round_path)

        best_idx = int(top_indices[0])
        print(
            "round=%d best=%d score=%.4f rr=%.4f cfr=%.4f"
            % (
                round_idx,
                best_idx,
                scores[best_idx],
                rr_scores[best_idx],
                cfr_eval_scores[best_idx],
            )
        )


if __name__ == "__main__":
    main()
