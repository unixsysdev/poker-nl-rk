import argparse
import inspect
import pathlib
import pickle
import time

import pyspiel
import numpy as np

from poker_rl.universal_poker import limit_holdem_params, nlhe_params


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
    validate_params(params)
    return pyspiel.load_game("universal_poker", params)


def _parse_int_list(value: str):
    return [int(x) for x in value.split()]


def validate_params(params):
    num_rounds = int(params["numRounds"])
    num_players = int(params["numPlayers"])

    def check_len(name):
        values = params[name].split()
        if len(values) != num_rounds:
            raise ValueError(f"{name} must have {num_rounds} entries, got {len(values)}")

    check_len("numBoardCards")
    check_len("raiseSize")
    check_len("maxRaises")
    check_len("firstPlayer")

    first_players = _parse_int_list(params["firstPlayer"])
    for idx, value in enumerate(first_players):
        if value < 1 or value > num_players:
            raise ValueError(f"firstPlayer[{idx}] must be in [1, {num_players}], got {value}")


def make_solver(game, args):
    if args.algorithm == "deep_cfr":
        try:
            from open_spiel.python.algorithms import deep_cfr
        except ImportError as exc:
            raise ImportError(
                "deep_cfr is not available in this OpenSpiel build. "
                "Try --algorithm cfr_plus or outcome_sampling_mccfr."
            ) from exc

        solver_cls = deep_cfr.DeepCFRSolver
        sig = inspect.signature(solver_cls)
        kwargs = {
            "policy_network_layers": (args.hidden_size, args.hidden_size),
            "advantage_network_layers": (args.hidden_size, args.hidden_size),
            "num_iterations": args.iterations,
            "num_traversals": args.traversals,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "memory_capacity": args.memory_capacity,
            "num_players": game.num_players(),
        }
        kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return solver_cls(game, **kwargs)

    if args.algorithm in {"cfr", "cfr_plus"}:
        from open_spiel.python.algorithms import cfr

        return cfr.CFRSolver(game) if args.algorithm == "cfr" else cfr.CFRPlusSolver(game)

    if args.algorithm == "outcome_sampling_mccfr":
        from open_spiel.python.algorithms import outcome_sampling_mccfr

        return outcome_sampling_mccfr.OutcomeSamplingSolver(game)

    if args.algorithm == "external_sampling_mccfr":
        from open_spiel.python.algorithms import external_sampling_mccfr

        return external_sampling_mccfr.ExternalSamplingSolver(game)

    raise ValueError(f"Unsupported algorithm: {args.algorithm}")


def _step_solver(solver):
    if hasattr(solver, "iteration"):
        solver.iteration()
        return
    if hasattr(solver, "evaluate_and_update_policy"):
        solver.evaluate_and_update_policy()
        return
    if hasattr(solver, "solve"):
        solver.solve()
        return
    raise RuntimeError("Solver does not expose iteration/evaluate_and_update_policy/solve.")


class RandomPolicy:
    def action_probabilities(self, state):
        legal_actions = state.legal_actions(state.current_player())
        prob = 1.0 / len(legal_actions)
        return {action: prob for action in legal_actions}


def play_episode(game, policy_a, policy_b, rng):
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            actions, probs = zip(*state.chance_outcomes())
            action = int(rng.choice(actions, p=probs))
            state.apply_action(action)
            continue

        player = state.current_player()
        policy = policy_a if player == 0 else policy_b
        probs = policy.action_probabilities(state)
        actions, weights = zip(*probs.items())
        action = int(rng.choice(actions, p=weights))
        state.apply_action(action)

    return state.returns()


def evaluate_policy(game, policy, episodes, seed):
    rng = np.random.default_rng(seed)
    random_policy = RandomPolicy()
    returns = []
    for i in range(episodes):
        seat = i % 2
        if seat == 0:
            result = play_episode(game, policy, random_policy, rng)[0]
        else:
            result = play_episode(game, random_policy, policy, rng)[1]
        returns.append(result)
    return float(np.mean(returns))


def export_policy(game, policy, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = policy
    try:
        from open_spiel.python import policy as policy_lib

        obj = policy_lib.tabular_policy_from_policy(game, policy)
    except Exception:
        pass

    with path.open("wb") as f:
        pickle.dump(obj, f)


def apply_preset(args):
    if args.preset == "best_nlhe":
        args.config = "nlhe"
        args.num_ranks = 13
        args.num_suits = 4
        args.stack = "20000 20000"
        args.blind = "50 100"
        args.raise_size = "100 200 400 800"
        args.max_raises = "4 4 4 4"
        args.first_player = "1 1 1 1"
        args.betting = "nolimit"
        args.betting_abstraction = "fcpa"
    elif args.preset == "real_nlhe":
        args.config = "nlhe"
        args.num_ranks = 13
        args.num_suits = 4
        args.stack = "10000 10000"
        args.blind = "50 100"
        args.raise_size = "100 150 300 600"
        args.max_raises = "4 4 4 4"
        args.first_player = "1 1 1 1"
        args.betting = "nolimit"
        args.betting_abstraction = "fcpa"
    elif args.preset == "granular_nlhe":
        args.config = "nlhe"
        args.num_ranks = 13
        args.num_suits = 4
        args.stack = "20000 20000"
        args.blind = "50 100"
        args.raise_size = "100 150 250 400"
        args.max_raises = "6 6 6 6"
        args.first_player = "1 1 1 1"
        args.betting = "nolimit"
        args.betting_abstraction = "fcpa"


def train(args):
    apply_preset(args)
    print(
        "start_train "
        f"preset={args.preset or 'none'} algorithm={args.algorithm} "
        f"iterations={args.iterations} seed={args.seed}"
    )
    print(
        "game_config "
        f"config={args.config} num_ranks={args.num_ranks} num_suits={args.num_suits} "
        f"stack={args.stack} blind={args.blind}"
    )
    print(
        "betting_config "
        f"betting={args.betting or 'default'} "
        f"abstraction={args.betting_abstraction or 'default'} "
        f"raise_size={args.raise_size} max_raises={args.max_raises} "
        f"first_player={args.first_player}"
    )
    game = build_game(args)
    solver = make_solver(game, args)
    start = time.time()
    for iteration in range(1, args.iterations + 1):
        _step_solver(solver)

        if args.log_every and iteration % args.log_every == 0:
            elapsed = time.time() - start
            print(f"iter={iteration} elapsed={elapsed:.1f}s")

        if args.eval_every and iteration % args.eval_every == 0:
            policy = solver.average_policy() if hasattr(solver, "average_policy") else solver.policy
            score = evaluate_policy(game, policy, args.eval_episodes, args.seed + iteration)
            print(f"eval@{iteration} avg_return_vs_random={score:.4f}")

        if args.checkpoint_every and iteration % args.checkpoint_every == 0:
            policy = solver.average_policy() if hasattr(solver, "average_policy") else solver.policy
            checkpoint_path = pathlib.Path(args.checkpoint_dir) / f"policy_iter_{iteration:06d}.pkl"
            export_policy(game, policy, checkpoint_path)
            print(f"checkpoint_saved={checkpoint_path}")

    if hasattr(solver, "average_policy"):
        policy = solver.average_policy()
    else:
        policy = getattr(solver, "policy", None)

    print("Training complete.")
    print("Policy type:", type(policy))
    return policy


def main():
    parser = argparse.ArgumentParser(description="Deep CFR on universal_poker (LHE-like).")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--traversals", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--memory-capacity", type=int, default=200000)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--config", choices=["lhe", "nlhe"], default="lhe")
    parser.add_argument(
        "--algorithm",
        choices=[
            "deep_cfr",
            "cfr",
            "cfr_plus",
            "outcome_sampling_mccfr",
            "external_sampling_mccfr",
        ],
        default="deep_cfr",
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="experiments/universal_poker_mccfr")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preset",
        choices=["best_nlhe", "real_nlhe", "granular_nlhe"],
        default=None,
    )

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
    train(args)


if __name__ == "__main__":
    main()
