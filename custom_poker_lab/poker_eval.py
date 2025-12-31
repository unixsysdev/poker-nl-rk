from __future__ import annotations

import argparse
import copy
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_poker_lab.poker_env import EnvConfig, NoLimitHoldemEnv
from custom_poker_lab.poker_policy import PolicyConfig, TwoHeadPolicy


class RandomPolicy:
    def act(self, state):
        legal = state["legal_action_mask"]
        actions = [i for i, v in enumerate(legal) if v > 0]
        action_type = int(np.random.choice(actions))
        bet_frac = float(np.random.random())
        return action_type, bet_frac


class LBRPolicy:
    def __init__(self, rollouts: int, bet_fracs: list[float], seed: int = 0):
        self.rollouts = rollouts
        self.bet_fracs = bet_fracs
        self.rng = np.random.default_rng(seed)

    def _random_action(self, state):
        legal = state["legal_action_mask"]
        actions = [i for i, v in enumerate(legal) if v > 0]
        action_type = int(self.rng.choice(actions))
        bet_frac = float(self.rng.random())
        return action_type, bet_frac

    def act(self, env: NoLimitHoldemEnv, player_id: int):
        state = env.get_state(player_id)
        legal = state["legal_action_mask"]
        candidates = []
        if legal[0] > 0:
            candidates.append((0, 0.0))
        if legal[1] > 0:
            candidates.append((1, 0.0))
        if legal[2] > 0:
            for frac in self.bet_fracs:
                bet_frac = max(0.0, min(1.0, frac / 2.0))
                candidates.append((2, bet_frac))

        best_action = candidates[0]
        best_score = -1e18
        for action_type, bet_frac in candidates:
            score = 0.0
            for _ in range(self.rollouts):
                sim = copy.deepcopy(env)
                sim.step(action_type, bet_frac)
                while not sim.is_over():
                    sim_state = sim.get_state(sim.current_player)
                    sim_action, sim_bet = self._random_action(sim_state)
                    sim.step(sim_action, sim_bet)
                score += sim.get_payoffs()[player_id]
            avg = score / max(1, self.rollouts)
            if avg > best_score:
                best_score = avg
                best_action = (action_type, bet_frac)
        return best_action


def load_policy(path: str, device: str, obs_dim: int):
    state = torch.load(path, map_location=device)
    hidden_dim = state.get("hidden_dim", 256)
    config = PolicyConfig(obs_dim=obs_dim, hidden_dim=hidden_dim)
    policy = TwoHeadPolicy(config, device=device)
    policy.load_state_dict(state["model"])
    return policy


def evaluate(env, policy, opponent, episodes):
    returns = []
    for _ in range(episodes):
        state, player_id = env.reset()
        while not env.is_over():
            if player_id == 0:
                action_type, bet_frac, _, _, _, _ = policy.act(state, deterministic=True)
            else:
                if isinstance(opponent, LBRPolicy):
                    action_type, bet_frac = opponent.act(env, player_id)
                else:
                    action_type, bet_frac = opponent.act(state)
            state, player_id = env.step(action_type, bet_frac)
        returns.append(env.get_payoffs()[0])
    return float(np.mean(returns))


def _eval_worker(payload):
    policy_state, env_config, episodes, seed, opponent_type, lbr_rollouts, lbr_bet_fracs, hidden_dim = payload
    local_cfg = EnvConfig(**env_config.__dict__)
    local_cfg.seed = seed
    env = NoLimitHoldemEnv(local_cfg)
    policy = TwoHeadPolicy(PolicyConfig(obs_dim=env.obs_dim, hidden_dim=hidden_dim), device="cpu")
    policy.load_state_dict(policy_state)
    if opponent_type == "lbr":
        opponent = LBRPolicy(lbr_rollouts, lbr_bet_fracs, seed=seed + 7)
    else:
        opponent = RandomPolicy()
    return evaluate(env, policy, opponent, episodes)


def evaluate_parallel(
    policy_state,
    env_config,
    episodes,
    opponent_type="random",
    eval_parallel=1,
    lbr_rollouts=32,
    lbr_bet_fracs=None,
    hidden_dim=256,
):
    if lbr_bet_fracs is None:
        lbr_bet_fracs = [0.5, 1.0, 2.0]
    if eval_parallel <= 1:
        env = NoLimitHoldemEnv(env_config)
        policy = TwoHeadPolicy(PolicyConfig(obs_dim=env.obs_dim, hidden_dim=hidden_dim), device="cpu")
        policy.load_state_dict(policy_state)
        if opponent_type == "lbr":
            opponent = LBRPolicy(lbr_rollouts, lbr_bet_fracs, seed=env_config.seed + 7)
        else:
            opponent = RandomPolicy()
        return evaluate(env, policy, opponent, episodes)

    import multiprocessing as mp

    per_worker = max(1, episodes // eval_parallel)
    payloads = []
    for i in range(eval_parallel):
        payloads.append(
            (
                policy_state,
                env_config,
                per_worker,
                env_config.seed + i + 123,
                opponent_type,
                lbr_rollouts,
                lbr_bet_fracs,
                hidden_dim,
            )
        )
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=eval_parallel) as pool:
        scores = pool.map(_eval_worker, payloads)
    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser(description="Evaluate custom NLHE policy.")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
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
    parser.add_argument("--opponent", choices=["random", "lbr"], default="random")
    parser.add_argument("--lbr-rollouts", type=int, default=32)
    parser.add_argument("--lbr-bet-fracs", default="0.5,1.0,2.0")
    parser.add_argument("--eval-parallel", type=int, default=1)
    args = parser.parse_args()

    state = torch.load(args.policy, map_location="cpu")
    saved_cfg = state.get("config")
    if saved_cfg:
        config = EnvConfig(**saved_cfg)
        config.seed = args.seed
    else:
        config = EnvConfig(
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
    bet_fracs = [float(x) for x in args.lbr_bet_fracs.split(",") if x]
    state_dict = torch.load(args.policy, map_location="cpu")
    policy_state = state_dict["model"] if "model" in state_dict else state_dict
    hidden_dim = state_dict.get("hidden_dim", 256) if isinstance(state_dict, dict) else 256
    score = evaluate_parallel(
        policy_state,
        config,
        args.episodes,
        opponent_type=args.opponent,
        eval_parallel=args.eval_parallel,
        lbr_rollouts=args.lbr_rollouts,
        lbr_bet_fracs=bet_fracs,
        hidden_dim=hidden_dim,
    )
    print(f"avg_return={score:.4f}")


if __name__ == "__main__":
    main()
