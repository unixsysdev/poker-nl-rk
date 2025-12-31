from __future__ import annotations

import argparse
import pathlib
import sys
import copy

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


def _policy_act(policy: TwoHeadPolicy, obs: torch.Tensor, mask: torch.Tensor):
    type_logits, size_params, _ = policy.net(obs.unsqueeze(0))
    type_logits = type_logits.masked_fill(mask.unsqueeze(0) == 0, -1e9)
    action_type = torch.argmax(type_logits, dim=-1)[0]
    alpha = torch.nn.functional.softplus(size_params[:, 0]) + 1.0
    beta = torch.nn.functional.softplus(size_params[:, 1]) + 1.0
    bet_frac = (alpha / (alpha + beta)).clamp(0.0, 1.0)[0]
    return int(action_type.item()), float(bet_frac.item())


class LBRPolicy:
    def __init__(self, rollouts: int, bet_fracs: list[float], seed: int = 0):
        self.rollouts = rollouts
        self.bet_fracs = bet_fracs
        self.rng = np.random.default_rng(seed)

    def _random_action(self, mask: torch.Tensor):
        actions = [i for i, v in enumerate(mask.detach().cpu().numpy()) if v > 0]
        if not actions:
            return 1, 0.0
        return int(self.rng.choice(actions)), float(self.rng.random())

    def act(self, env: CudaNLHEEnv, player_id: int, policy: TwoHeadPolicy):
        obs, mask, _ = env.get_obs()
        legal = mask[0]
        candidates = []
        if legal[0] > 0:
            candidates.append((0, 0.0))
        if legal[1] > 0:
            candidates.append((1, 0.0))
        if legal[2] > 0:
            for frac in self.bet_fracs:
                bet_frac = max(0.0, min(1.0, frac))
                candidates.append((2, bet_frac))

        best_action = candidates[0]
        best_score = -1e18
        for action_type, bet_frac in candidates:
            score = 0.0
            for _ in range(self.rollouts):
                sim = copy.deepcopy(env)
                sim.step(
                    torch.tensor([action_type], device=env.device, dtype=torch.int64),
                    torch.tensor([bet_frac], device=env.device, dtype=torch.float32),
                )
                while not bool(sim.episode_over[0].item()):
                    sim_obs, sim_mask, sim_player = sim.get_obs()
                    pid = int(sim_player[0].item())
                    if pid == 0:
                        a_type, b_frac = _policy_act(policy, sim_obs[0], sim_mask[0])
                    else:
                        a_type, b_frac = self._random_action(sim_mask[0])
                    sim.step(
                        torch.tensor([a_type], device=env.device, dtype=torch.int64),
                        torch.tensor([b_frac], device=env.device, dtype=torch.float32),
                    )
                score += sim.get_payoffs()[0, player_id].item()
            avg = score / max(1, self.rollouts)
            if avg > best_score:
                best_score = avg
                best_action = (action_type, bet_frac)
        return best_action


class DepthLimitedBRPolicy:
    def __init__(
        self,
        depth: int,
        rollouts: int,
        bet_fracs: list[float],
        seed: int = 0,
        other_samples: int = 1,
    ):
        self.depth = depth
        self.rollouts = rollouts
        self.bet_fracs = bet_fracs
        self.rng = np.random.default_rng(seed)
        self.other_samples = other_samples

    def _random_action(self, mask: torch.Tensor):
        actions = [i for i, v in enumerate(mask.detach().cpu().numpy()) if v > 0]
        if not actions:
            return 1, 0.0
        return int(self.rng.choice(actions)), float(self.rng.random())

    def _rollout_value(self, env: CudaNLHEEnv, policy: TwoHeadPolicy):
        score = 0.0
        for _ in range(self.rollouts):
            sim = copy.deepcopy(env)
            while not bool(sim.episode_over[0].item()):
                sim_obs, sim_mask, sim_player = sim.get_obs()
                pid = int(sim_player[0].item())
                if pid == 0:
                    a_type, b_frac = _policy_act(policy, sim_obs[0], sim_mask[0])
                else:
                    a_type, b_frac = self._random_action(sim_mask[0])
                sim.step(
                    torch.tensor([a_type], device=env.device, dtype=torch.int64),
                    torch.tensor([b_frac], device=env.device, dtype=torch.float32),
                )
            score += sim.get_payoffs()[0, 0].item()
        return score / max(1, self.rollouts)

    def _search(self, env: CudaNLHEEnv, policy: TwoHeadPolicy, br_player: int, depth: int):
        if bool(env.episode_over[0].item()):
            return float(env.get_payoffs()[0, 0].item())
        if depth <= 0:
            return self._rollout_value(env, policy)

        obs, mask, player = env.get_obs()
        pid = int(player[0].item())
        legal = mask[0]
        if pid == br_player:
            candidates = []
            if legal[0] > 0:
                candidates.append((0, 0.0))
            if legal[1] > 0:
                candidates.append((1, 0.0))
            if legal[2] > 0:
                for frac in self.bet_fracs:
                    bet_frac = max(0.0, min(1.0, frac))
                    candidates.append((2, bet_frac))
            best = float("inf")
            for action_type, bet_frac in candidates:
                sim = copy.deepcopy(env)
                sim.step(
                    torch.tensor([action_type], device=env.device, dtype=torch.int64),
                    torch.tensor([bet_frac], device=env.device, dtype=torch.float32),
                )
                val = self._search(sim, policy, br_player, depth - 1)
                best = min(best, val)
            return best
        if pid == 0:
            action_type, bet_frac = _policy_act(policy, obs[0], legal)
            sim = copy.deepcopy(env)
            sim.step(
                torch.tensor([action_type], device=env.device, dtype=torch.int64),
                torch.tensor([bet_frac], device=env.device, dtype=torch.float32),
            )
            return self._search(sim, policy, br_player, depth - 1)

        total = 0.0
        for _ in range(self.other_samples):
            action_type, bet_frac = self._random_action(legal)
            sim = copy.deepcopy(env)
            sim.step(
                torch.tensor([action_type], device=env.device, dtype=torch.int64),
                torch.tensor([bet_frac], device=env.device, dtype=torch.float32),
            )
            total += self._search(sim, policy, br_player, depth - 1)
        return total / max(1, self.other_samples)

    def act(self, env: CudaNLHEEnv, player_id: int, policy: TwoHeadPolicy):
        obs, mask, _ = env.get_obs()
        legal = mask[0]
        candidates = []
        if legal[0] > 0:
            candidates.append((0, 0.0))
        if legal[1] > 0:
            candidates.append((1, 0.0))
        if legal[2] > 0:
            for frac in self.bet_fracs:
                bet_frac = max(0.0, min(1.0, frac))
                candidates.append((2, bet_frac))
        best_action = candidates[0]
        best_score = float("inf")
        for action_type, bet_frac in candidates:
            sim = copy.deepcopy(env)
            sim.step(
                torch.tensor([action_type], device=env.device, dtype=torch.int64),
                torch.tensor([bet_frac], device=env.device, dtype=torch.float32),
            )
            val = self._search(sim, policy, player_id, self.depth - 1)
            if val < best_score:
                best_score = val
                best_action = (action_type, bet_frac)
        return best_action


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
    parser.add_argument("--cpu-eval-workers", type=int, default=0)
    parser.add_argument("--cpu-eval-min-batch", type=int, default=8)
    parser.add_argument("--opponent", choices=["random", "lbr", "dlbr", "proxy"], default="random")
    parser.add_argument("--lbr-rollouts", type=int, default=16)
    parser.add_argument("--lbr-bet-fracs", default="0.25,0.5,1.0")
    parser.add_argument("--br-depth", type=int, default=2)
    parser.add_argument("--br-other-samples", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=0)
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
        cpu_eval_workers=args.cpu_eval_workers,
        cpu_eval_min_batch=args.cpu_eval_min_batch,
    )
    env = CudaNLHEEnv(env_config)
    policy = TwoHeadPolicy(PolicyConfig(obs_dim=env.obs_dim, hidden_dim=hidden_dim), device=args.device)
    policy.load_state_dict(state["model"])
    bet_fracs = [float(x) for x in args.lbr_bet_fracs.split(",") if x]
    random_opponent = RandomPolicy(seed=args.seed + 7)
    lbr = LBRPolicy(args.lbr_rollouts, bet_fracs, seed=args.seed + 11)
    dlbr = DepthLimitedBRPolicy(
        args.br_depth, args.lbr_rollouts, bet_fracs, seed=args.seed + 13, other_samples=args.br_other_samples
    )

    def run_eval(opponent_type: str) -> float:
        returns = []
        for idx in range(args.episodes):
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
                    if opponent_type == "lbr":
                        a_type, b_frac = lbr.act(env, int(current[0].item()), policy)
                    elif opponent_type == "dlbr":
                        a_type, b_frac = dlbr.act(env, int(current[0].item()), policy)
                    else:
                        a_type, b_frac = random_opponent.act(mask[0].detach().cpu().numpy())
                    action_types = torch.tensor([a_type], device=env.device, dtype=torch.int64)
                    bet_fracs = torch.tensor([b_frac], device=env.device, dtype=torch.float32)
                env.step(action_types, bet_fracs)
            returns.append(float(env.get_payoffs()[0, 0].item()))
            if args.log_every and (idx + 1) % args.log_every == 0:
                print(
                    f"eval_progress opponent={opponent_type} episode={idx + 1}/{args.episodes}",
                    flush=True,
                )
        return float(np.mean(returns))

    if args.opponent == "proxy":
        random_score = run_eval("random")
        lbr_score = run_eval("lbr")
        score = min(random_score, lbr_score)
    else:
        score = run_eval(args.opponent)
    print(f"avg_return={score:.4f}")


if __name__ == "__main__":
    main()
