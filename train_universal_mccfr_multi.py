import argparse
import json
import multiprocessing as mp
import os
import pathlib
import sys
import time

import train_universal_deep_cfr as trainer


def parse_seeds(args):
    if args.seeds:
        return [int(x) for x in args.seeds.split(",") if x.strip()]
    return [args.base_seed + i for i in range(args.num_seeds)]


def build_args_dict(args):
    return {
        "iterations": args.iterations,
        "traversals": args.traversals,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "memory_capacity": args.memory_capacity,
        "hidden_size": args.hidden_size,
        "config": args.config,
        "algorithm": args.algorithm,
        "log_every": args.log_every,
        "eval_every": args.eval_every,
        "eval_episodes": args.eval_episodes,
        "checkpoint_every": args.checkpoint_every,
        "checkpoint_dir": args.checkpoint_dir,
        "seed": args.base_seed,
        "preset": None if args.preset == "none" else args.preset,
        "num_ranks": args.num_ranks,
        "num_suits": args.num_suits,
        "stack": args.stack,
        "blind": args.blind,
        "raise_size": args.raise_size,
        "max_raises": args.max_raises,
        "first_player": args.first_player,
        "betting": args.betting,
        "betting_abstraction": args.betting_abstraction,
    }


def train_one(payload):
    args_dict, seed, log_dir, redirect_logs = payload
    args = argparse.Namespace(**args_dict)
    args.seed = seed
    args.checkpoint_dir = str(pathlib.Path(args.checkpoint_dir) / f"seed_{seed}")

    log_path = None
    if log_dir and redirect_logs:
        log_dir = pathlib.Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"seed_{seed}.log"
        sys.stdout = log_path.open("w", buffering=1)
        sys.stderr = sys.stdout

    start = time.time()
    trainer.train(args)
    elapsed = time.time() - start
    return {
        "seed": seed,
        "checkpoint_dir": args.checkpoint_dir,
        "elapsed_sec": elapsed,
        "log_path": str(log_path) if log_path else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Parallel MCCFR runs for universal_poker.")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--num-seeds", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--parallel", type=int, default=None)
    parser.add_argument("--log-dir", default="experiments/universal_poker_mccfr_multi/logs")
    parser.add_argument("--no-log-redirect", action="store_true")
    parser.add_argument("--summary-path", default=None)
    parser.add_argument(
        "--preset",
        choices=["best_nlhe", "real_nlhe", "granular_nlhe", "none"],
        default="best_nlhe",
    )

    parser.add_argument("--iterations", type=int, default=5000000)
    parser.add_argument("--traversals", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--memory-capacity", type=int, default=200000)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--config", choices=["lhe", "nlhe"], default="nlhe")
    parser.add_argument(
        "--algorithm",
        choices=[
            "deep_cfr",
            "cfr",
            "cfr_plus",
            "outcome_sampling_mccfr",
            "external_sampling_mccfr",
        ],
        default="outcome_sampling_mccfr",
    )
    parser.add_argument("--log-every", type=int, default=100000)
    parser.add_argument("--eval-every", type=int, default=500000)
    parser.add_argument("--eval-episodes", type=int, default=5000)
    parser.add_argument("--checkpoint-every", type=int, default=500000)
    parser.add_argument("--checkpoint-dir", default="experiments/universal_poker_mccfr_multi")

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
    seeds = parse_seeds(args)
    if not seeds:
        raise ValueError("No seeds specified.")

    redirect_logs = not args.no_log_redirect
    parallel = args.parallel or min(len(seeds), max(1, (os.cpu_count() or 4) // 4))
    args_dict = build_args_dict(args)
    log_dir = args.log_dir if redirect_logs else None
    payloads = [(args_dict, seed, log_dir, redirect_logs) for seed in seeds]

    print(
        "multi_seed_start "
        f"seeds={seeds} parallel={parallel} "
        f"checkpoint_dir={args.checkpoint_dir} "
        f"redirect_logs={redirect_logs}"
    )
    if redirect_logs and args.log_dir:
        print(f"log_dir={args.log_dir}")
        for seed in seeds:
            print(f"seed_log=seed_{seed}.log")

    ctx = mp.get_context("spawn")
    results = []
    with ctx.Pool(processes=parallel) as pool:
        for idx, result in enumerate(pool.imap_unordered(train_one, payloads), start=1):
            results.append(result)
            print(
                "seed_done "
                f"{idx}/{len(seeds)} seed={result['seed']} "
                f"elapsed_sec={result['elapsed_sec']:.1f}"
            )

    summary_path = (
        pathlib.Path(args.summary_path)
        if args.summary_path
        else pathlib.Path(args.checkpoint_dir) / "summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"summary_saved={summary_path}")


if __name__ == "__main__":
    main()
