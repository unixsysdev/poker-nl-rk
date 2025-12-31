import argparse
import multiprocessing as mp
import pathlib
import pickle
import tempfile

import numpy as np
import pyspiel

from poker_rl.universal_poker import limit_holdem_params, nlhe_params


class RandomPolicy:
    def action_probabilities(self, state):
        legal_actions = state.legal_actions(state.current_player())
        prob = 1.0 / len(legal_actions)
        return {action: prob for action in legal_actions}


class LocalBestResponsePolicy:
    def __init__(self, game, target_policy, player_id, rollouts=32, seed=0):
        self.game = game
        self.target_policy = target_policy
        self.player_id = player_id
        self.rollouts = max(1, int(rollouts))
        self.rng = np.random.default_rng(seed)

    def action_probabilities(self, state):
        legal_actions = state.legal_actions(state.current_player())
        if len(legal_actions) == 1:
            return {legal_actions[0]: 1.0}

        # Simple rollout-based best response (proxy), not an exact BR.
        best_action = legal_actions[0]
        best_value = -float("inf")
        for action in legal_actions:
            total = 0.0
            for _ in range(self.rollouts):
                total += self._rollout_from(state, action)
            avg = total / self.rollouts
            if avg > best_value:
                best_value = avg
                best_action = action

        return {best_action: 1.0}

    def _rollout_from(self, state, action):
        st = state.clone()
        st.apply_action(action)
        while not st.is_terminal():
            if st.is_chance_node():
                actions, probs = zip(*st.chance_outcomes())
                st.apply_action(int(self.rng.choice(actions, p=probs)))
                continue

            player = st.current_player()
            if player == self.player_id:
                legal_actions = st.legal_actions(player)
                st.apply_action(int(self.rng.choice(legal_actions)))
            else:
                probs = self.target_policy.action_probabilities(st)
                actions, weights = zip(*probs.items())
                st.apply_action(int(self.rng.choice(actions, p=weights)))

        return st.returns()[self.player_id]

def build_game(args):
    if args.config == "nlhe":
        params = nlhe_params(
            num_ranks=args.num_ranks,
            num_suits=args.num_suits,
            stack=args.stack,
            blind=args.blind,
            betting_abstraction=args.betting_abstraction,
            raise_size=args.raise_size,
            max_raises=args.max_raises,
            first_player=args.first_player,
        )
    else:
        params = limit_holdem_params(
            num_ranks=args.num_ranks,
            num_suits=args.num_suits,
            stack=args.stack,
            blind=args.blind,
            raise_size=args.raise_size,
            max_raises=args.max_raises,
            first_player=args.first_player,
        )
    if args.betting is not None:
        params["betting"] = args.betting
    if args.betting_abstraction is not None:
        params["bettingAbstraction"] = args.betting_abstraction
    return pyspiel.load_game("universal_poker", params)


def load_policy(path):
    with pathlib.Path(path).open("rb") as f:
        return pickle.load(f)


def export_policy(game, policy, path):
    obj = policy
    try:
        from open_spiel.python import policy as policy_lib

        obj = policy_lib.tabular_policy_from_policy(game, policy)
    except Exception:
        pass

    with pathlib.Path(path).open("wb") as f:
        pickle.dump(obj, f)


def _step_solver(solver):
    if hasattr(solver, "evaluate_and_update_policy"):
        solver.evaluate_and_update_policy()
        return
    if hasattr(solver, "iteration"):
        solver.iteration()
        return
    if hasattr(solver, "solve"):
        solver.solve()
        return
    raise RuntimeError("Solver does not expose iteration/evaluate_and_update_policy/solve.")


def play_episode(game, policy_a, policy_b, rng):
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            actions, probs = zip(*state.chance_outcomes())
            state.apply_action(int(rng.choice(actions, p=probs)))
            continue

        player = state.current_player()
        policy = policy_a if player == 0 else policy_b
        probs = policy.action_probabilities(state)
        actions, weights = zip(*probs.items())
        action = int(rng.choice(actions, p=weights))
        state.apply_action(action)

    return state.returns()


def evaluate(game, policy, opponent, episodes, seed, opponent_alt=None):
    rng = np.random.default_rng(seed)
    total = 0.0
    for i in range(episodes):
        seat = i % 2
        if seat == 0:
            total += play_episode(game, policy, opponent, rng)[0]
        else:
            alt = opponent_alt or opponent
            total += play_episode(game, alt, policy, rng)[1]
    return total, episodes


def _eval_worker(payload):
    args_dict, policy_path, opponent_mode, opponent_path, lbr_rollouts, episodes, seed = payload
    args = argparse.Namespace(**args_dict)
    game = build_game(args)
    policy = load_policy(policy_path)
    opponent_alt = None
    if opponent_mode == "policy":
        opponent = load_policy(opponent_path)
    elif opponent_mode == "lbr":
        opponent = LocalBestResponsePolicy(
            game,
            target_policy=policy,
            player_id=1,
            rollouts=lbr_rollouts,
            seed=seed,
        )
        opponent_alt = LocalBestResponsePolicy(
            game,
            target_policy=policy,
            player_id=0,
            rollouts=lbr_rollouts,
            seed=seed + 9999,
        )
    else:
        opponent = RandomPolicy()
    return evaluate(game, policy, opponent, episodes, seed, opponent_alt=opponent_alt)


def main():
    parser = argparse.ArgumentParser(description="Evaluate universal_poker policies.")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--opponent", choices=["random", "policy", "cfr", "lbr"], default="random")
    parser.add_argument("--opponent-policy", default=None)
    parser.add_argument("--cfr-iterations", type=int, default=1000)
    parser.add_argument(
        "--cfr-algorithm",
        choices=["cfr", "cfr_plus", "outcome_sampling_mccfr", "external_sampling_mccfr"],
        default="cfr",
    )
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-parallel", type=int, default=1)
    parser.add_argument("--lbr-rollouts", type=int, default=32)

    parser.add_argument("--config", choices=["lhe", "nlhe"], default="lhe")
    parser.add_argument("--num-ranks", type=int, default=6)
    parser.add_argument("--num-suits", type=int, default=4)
    parser.add_argument("--stack", default="2000 2000")
    parser.add_argument("--blind", default="50 100")
    parser.add_argument("--raise-size", default="100 100 200 200")
    parser.add_argument("--max-raises", default="4 4 4 4")
    parser.add_argument("--first-player", default="1 1 1 1")
    parser.add_argument("--betting", choices=["limit", "nolimit"], default=None)
    parser.add_argument("--betting-abstraction", default=None)
    args = parser.parse_args()

    game = build_game(args)
    policy = load_policy(args.policy)

    opponent_mode = args.opponent
    opponent_policy_path = args.opponent_policy
    temp_policy_path = None
    opponent_alt = None

    if args.opponent == "policy":
        if not args.opponent_policy:
            raise ValueError("--opponent-policy is required when --opponent=policy")
        opponent = load_policy(args.opponent_policy)
    elif args.opponent == "cfr":
        if args.cfr_algorithm in {"cfr", "cfr_plus"}:
            from open_spiel.python.algorithms import cfr

            solver = cfr.CFRSolver(game) if args.cfr_algorithm == "cfr" else cfr.CFRPlusSolver(game)
        elif args.cfr_algorithm == "outcome_sampling_mccfr":
            from open_spiel.python.algorithms import outcome_sampling_mccfr

            solver = outcome_sampling_mccfr.OutcomeSamplingSolver(game)
        else:
            from open_spiel.python.algorithms import external_sampling_mccfr

            solver = external_sampling_mccfr.ExternalSamplingSolver(game)

        for _ in range(args.cfr_iterations):
            _step_solver(solver)
        opponent = solver.average_policy()
    elif args.opponent == "lbr":
        opponent = LocalBestResponsePolicy(
            game,
            target_policy=policy,
            player_id=1,
            rollouts=args.lbr_rollouts,
            seed=args.seed,
        )
        opponent_alt = LocalBestResponsePolicy(
            game,
            target_policy=policy,
            player_id=0,
            rollouts=args.lbr_rollouts,
            seed=args.seed + 9999,
        )
    else:
        opponent = RandomPolicy()

    if args.opponent == "cfr" and args.eval_parallel > 1:
        temp_policy_path = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False).name
        export_policy(game, opponent, temp_policy_path)
        opponent_mode = "policy"
        opponent_policy_path = temp_policy_path
    elif args.opponent == "lbr" and args.eval_parallel > 1:
        opponent_mode = "lbr"

    if args.eval_parallel > 1 and opponent_mode in {"random", "policy"}:
        episodes_per_worker = [args.episodes // args.eval_parallel] * args.eval_parallel
        for i in range(args.episodes % args.eval_parallel):
            episodes_per_worker[i] += 1

        payloads = []
        args_dict = vars(args)
        for idx, count in enumerate(episodes_per_worker):
            if count == 0:
                continue
            payloads.append(
                (
                    args_dict,
                    args.policy,
                    opponent_mode,
                    opponent_policy_path,
                    args.lbr_rollouts,
                    count,
                    args.seed + idx,
                )
            )
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.eval_parallel) as pool:
            results = pool.map(_eval_worker, payloads)
        total = sum(x[0] for x in results)
        count = sum(x[1] for x in results)
        avg_return = total / count
    elif args.eval_parallel > 1 and opponent_mode == "lbr":
        episodes_per_worker = [args.episodes // args.eval_parallel] * args.eval_parallel
        for i in range(args.episodes % args.eval_parallel):
            episodes_per_worker[i] += 1

        payloads = []
        args_dict = vars(args)
        for idx, count in enumerate(episodes_per_worker):
            if count == 0:
                continue
            payloads.append(
                (
                    args_dict,
                    args.policy,
                    opponent_mode,
                    opponent_policy_path,
                    args.lbr_rollouts,
                    count,
                    args.seed + idx,
                )
            )
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.eval_parallel) as pool:
            results = pool.map(_eval_worker, payloads)
        total = sum(x[0] for x in results)
        count = sum(x[1] for x in results)
        avg_return = total / count
    else:
        total, count = evaluate(
            game,
            policy,
            opponent,
            args.episodes,
            args.seed,
            opponent_alt=opponent_alt,
        )
        avg_return = total / count

    if temp_policy_path is not None:
        pathlib.Path(temp_policy_path).unlink(missing_ok=True)

    print(f"avg_return={avg_return:.4f}")


if __name__ == "__main__":
    main()
