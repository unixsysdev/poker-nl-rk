import argparse
import multiprocessing as mp
import pathlib
import re

import eval_universal_policy as evaler


CHECKPOINT_RE = re.compile(r"policy_iter_(\d+)\.pkl")


def find_latest_checkpoint(seed_dir, checkpoint_iter=None):
    if checkpoint_iter is not None:
        path = seed_dir / f"policy_iter_{checkpoint_iter:06d}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {path}")
        return path

    best_iter = -1
    best_path = None
    for path in seed_dir.glob("policy_iter_*.pkl"):
        match = CHECKPOINT_RE.match(path.name)
        if not match:
            continue
        iter_id = int(match.group(1))
        if iter_id > best_iter:
            best_iter = iter_id
            best_path = path
    if best_path is None:
        raise FileNotFoundError(f"No checkpoints found in {seed_dir}")
    return best_path


def make_opponent(game, policy, args, seed_offset=0):
    opponent_alt = None
    if args.opponent == "policy":
        opponent = evaler.load_policy(args.opponent_policy)
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
            evaler._step_solver(solver)
        opponent = solver.average_policy()
    elif args.opponent == "lbr":
        opponent = evaler.LocalBestResponsePolicy(
            game,
            target_policy=policy,
            player_id=1,
            rollouts=args.lbr_rollouts,
            seed=args.seed + seed_offset,
        )
        opponent_alt = evaler.LocalBestResponsePolicy(
            game,
            target_policy=policy,
            player_id=0,
            rollouts=args.lbr_rollouts,
            seed=args.seed + seed_offset + 9999,
        )
    else:
        opponent = evaler.RandomPolicy()
    return opponent, opponent_alt


def evaluate_seed(payload):
    args_dict, seed_dir, checkpoint_iter, seed_offset = payload
    args = argparse.Namespace(**args_dict)
    seed_dir = pathlib.Path(seed_dir)
    checkpoint = find_latest_checkpoint(seed_dir, checkpoint_iter)
    game = evaler.build_game(args)
    policy = evaler.load_policy(checkpoint)
    opponent, opponent_alt = make_opponent(game, policy, args, seed_offset=seed_offset)
    total, count = evaler.evaluate(
        game,
        policy,
        opponent,
        args.episodes,
        args.seed + seed_offset,
        opponent_alt=opponent_alt,
    )
    return {
        "seed_dir": str(seed_dir),
        "checkpoint": str(checkpoint),
        "avg_return": total / count,
    }


def main():
    parser = argparse.ArgumentParser(description="Select best MCCFR seed by evaluation score.")
    parser.add_argument("--root-dir", default="experiments/universal_poker_mccfr_multi")
    parser.add_argument("--checkpoint-iter", type=int, default=None)
    parser.add_argument("--parallel", type=int, default=4)

    parser.add_argument("--opponent", choices=["random", "policy", "cfr", "lbr"], default="random")
    parser.add_argument("--opponent-policy", default=None)
    parser.add_argument("--cfr-iterations", type=int, default=1000)
    parser.add_argument(
        "--cfr-algorithm",
        choices=["cfr", "cfr_plus", "outcome_sampling_mccfr", "external_sampling_mccfr"],
        default="outcome_sampling_mccfr",
    )
    parser.add_argument("--lbr-rollouts", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--config", choices=["lhe", "nlhe"], default="nlhe")
    parser.add_argument("--num-ranks", type=int, default=13)
    parser.add_argument("--num-suits", type=int, default=4)
    parser.add_argument("--stack", default="20000 20000")
    parser.add_argument("--blind", default="50 100")
    parser.add_argument("--raise-size", default="100 200 400 800")
    parser.add_argument("--max-raises", default="4 4 4 4")
    parser.add_argument("--first-player", default="1 1 1 1")
    parser.add_argument("--betting", choices=["limit", "nolimit"], default=None)
    parser.add_argument("--betting-abstraction", default="fcpa")
    args = parser.parse_args()

    if args.opponent == "policy" and not args.opponent_policy:
        raise ValueError("--opponent-policy is required when --opponent=policy")

    root_dir = pathlib.Path(args.root_dir)
    seed_dirs = sorted([p for p in root_dir.glob("seed_*") if p.is_dir()])
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* dirs found in {root_dir}")

    args_dict = vars(args)
    payloads = [
        (args_dict, seed_dir, args.checkpoint_iter, idx)
        for idx, seed_dir in enumerate(seed_dirs)
    ]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(args.parallel, len(payloads))) as pool:
        results = pool.map(evaluate_seed, payloads)

    results_sorted = sorted(results, key=lambda x: x["avg_return"], reverse=True)
    for entry in results_sorted:
        print(f"seed_dir={entry['seed_dir']} avg_return={entry['avg_return']:.4f}")

    best = results_sorted[0]
    print(
        "best_seed_dir="
        f"{best['seed_dir']} best_checkpoint={best['checkpoint']} "
        f"best_avg_return={best['avg_return']:.4f}"
    )


if __name__ == "__main__":
    main()
