from __future__ import annotations

import argparse
import copy
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_poker_lab.poker_policy import PolicyConfig, TwoHeadPolicy
from custom_poker_lab.prod_lab.vector_env import VectorEnvConfig, VectorNLHEEnv


class RandomPolicy:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, mask):
        actions = [i for i, v in enumerate(mask) if v > 0]
        if not actions:
            return 1, 0.0
        action_type = int(self.rng.choice(actions))
        bet_frac = float(self.rng.random())
        return action_type, bet_frac


def _policy_act(policy: TwoHeadPolicy, obs: np.ndarray, mask: np.ndarray, deterministic: bool = True):
    state = {"obs": obs, "legal_action_mask": mask}
    action_type, bet_frac, *_ = policy.act(state, deterministic=deterministic)
    return action_type, bet_frac


class LBRPolicy:
    def __init__(self, rollouts: int, bet_fracs: list[float], seed: int = 0):
        self.rollouts = rollouts
        self.bet_fracs = bet_fracs
        self.rng = np.random.default_rng(seed)

    def act(self, env: VectorNLHEEnv, player_id: int, policy: TwoHeadPolicy):
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
                sim.step(np.array([action_type]), np.array([bet_frac]))
                while not sim.episode_over[0]:
                    sim_obs, sim_mask, sim_player = sim.get_obs()
                    pid = int(sim_player[0])
                    if pid == 0:
                        a_type, b_frac = _policy_act(policy, sim_obs[0], sim_mask[0], deterministic=True)
                    else:
                        actions = [i for i, v in enumerate(sim_mask[0]) if v > 0]
                        if actions:
                            a_type = int(self.rng.choice(actions))
                            b_frac = float(self.rng.random())
                        else:
                            a_type = 1
                            b_frac = 0.0
                    sim.step(np.array([a_type]), np.array([b_frac]))
                score += sim.get_payoffs()[0, player_id]
            avg = score / max(1, self.rollouts)
            if avg > best_score:
                best_score = avg
                best_action = (action_type, bet_frac)
        return best_action


def load_policy(policy_ref, obs_dim: int, device: str = "cpu"):
    if isinstance(policy_ref, str):
        state = torch.load(policy_ref, map_location=device)
        state_dict = state["model"] if "model" in state else state
        hidden_dim = state.get("hidden_dim", 256) if isinstance(state, dict) else 256
    else:
        state_dict = policy_ref
        hidden_dim = 256
    policy = TwoHeadPolicy(PolicyConfig(obs_dim=obs_dim, hidden_dim=hidden_dim), device=device)
    policy.load_state_dict(state_dict)
    return policy


def _evaluate_single(
    policy_ref,
    env_config: VectorEnvConfig,
    episodes: int,
    opponent: str,
    lbr_rollouts: int,
    lbr_bet_fracs: list[float],
):
    if lbr_bet_fracs is None:
        lbr_bet_fracs = [0.25, 0.5, 1.0]
    config = VectorEnvConfig(**env_config.__dict__)
    config.batch_size = 1
    env = VectorNLHEEnv(config)
    policy = load_policy(policy_ref, env.obs_dim, device="cpu")
    random_policy = RandomPolicy(seed=config.seed + 7)
    lbr = LBRPolicy(lbr_rollouts, lbr_bet_fracs, seed=config.seed + 11)

    returns = []
    for _ in range(episodes):
        env.reset()
        while not env.episode_over[0]:
            obs, mask, player = env.get_obs()
            pid = int(player[0])
            if pid == 0:
                action_type, bet_frac = _policy_act(policy, obs[0], mask[0], deterministic=True)
            else:
                if opponent == "lbr":
                    action_type, bet_frac = lbr.act(env, pid, policy)
                else:
                    action_type, bet_frac = random_policy.act(obs[0], mask[0])
            env.step(np.array([action_type]), np.array([bet_frac]))
        returns.append(env.get_payoffs()[0, 0])
    return float(np.mean(returns))


def evaluate(
    policy_ref,
    env_config: VectorEnvConfig,
    episodes: int,
    opponent: str = "random",
    lbr_rollouts: int = 32,
    lbr_bet_fracs: list[float] | None = None,
):
    if lbr_bet_fracs is None:
        lbr_bet_fracs = [0.25, 0.5, 1.0]
    if opponent == "proxy":
        random_score = _evaluate_single(
            policy_ref,
            env_config,
            episodes,
            opponent="random",
            lbr_rollouts=lbr_rollouts,
            lbr_bet_fracs=lbr_bet_fracs,
        )
        lbr_score = _evaluate_single(
            policy_ref,
            env_config,
            episodes,
            opponent="lbr",
            lbr_rollouts=lbr_rollouts,
            lbr_bet_fracs=lbr_bet_fracs,
        )
        return float(min(random_score, lbr_score))
    return _evaluate_single(
        policy_ref,
        env_config,
        episodes,
        opponent=opponent,
        lbr_rollouts=lbr_rollouts,
        lbr_bet_fracs=lbr_bet_fracs,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate production vectorized NLHE policy.")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--opponent", choices=["random", "lbr", "proxy"], default="random")
    parser.add_argument("--lbr-rollouts", type=int, default=32)
    parser.add_argument("--lbr-bet-fracs", default="0.25,0.5,1.0")
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
    args = parser.parse_args()

    bet_fracs = [float(x) for x in args.lbr_bet_fracs.split(",") if x]
    env_config = VectorEnvConfig(
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
    )
    score = evaluate(
        args.policy,
        env_config,
        args.episodes,
        opponent=args.opponent,
        lbr_rollouts=args.lbr_rollouts,
        lbr_bet_fracs=bet_fracs,
    )
    print(f"avg_return={score:.4f}")


if __name__ == "__main__":
    main()
