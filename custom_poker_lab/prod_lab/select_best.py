from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

from custom_poker_lab.prod_lab.vector_env import VectorEnvConfig
from custom_poker_lab.prod_lab import eval as evaler


def _score_checkpoint(payload):
    path, env_config_dict, args = payload
    env_config = VectorEnvConfig(**env_config_dict)
    score = evaler.evaluate(
        str(path),
        env_config,
        args.episodes,
        opponent=args.opponent,
        lbr_rollouts=args.lbr_rollouts,
        lbr_bet_fracs=args.lbr_bet_fracs,
        br_depth=args.br_depth,
        br_other_samples=args.br_other_samples,
    )
    return str(path), float(score)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the best checkpoint from a directory.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--glob", default="policy_ep_*.pt")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--latest", action="store_true", help="Select the latest checkpoint without evaluation.")
    parser.add_argument("--opponent", choices=["random", "lbr", "dlbr", "proxy"], default="proxy")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--write-best", default="best_checkpoint.txt")
    parser.add_argument("--nested-eval-workers", action="store_true")
    parser.add_argument("--lbr-rollouts", type=int, default=32)
    parser.add_argument("--lbr-bet-fracs", default="0.25,0.5,1.0")
    parser.add_argument("--br-depth", type=int, default=2)
    parser.add_argument("--br-other-samples", type=int, default=1)
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
    parser.add_argument("--cpu-eval-workers", type=int, default=0)
    parser.add_argument("--cpu-eval-min-batch", type=int, default=8)
    args = parser.parse_args()

    args.lbr_bet_fracs = [float(x) for x in args.lbr_bet_fracs.split(",") if x]

    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.exists():
        raise SystemExit(f"checkpoint_dir_not_found={ckpt_dir}")

    if args.recursive:
        paths = sorted(ckpt_dir.rglob(args.glob))
    else:
        paths = sorted(ckpt_dir.glob(args.glob))

    if not paths:
        raise SystemExit(f"no_checkpoints_match={args.glob}")

    if args.latest:
        def _episode_key(path: Path) -> Tuple[int, float]:
            stem = path.stem
            digits = "".join(ch for ch in stem if ch.isdigit())
            return (int(digits) if digits else -1, path.stat().st_mtime)

        best_path = max(paths, key=_episode_key)
        print(f"best_checkpoint={best_path}")
        if args.write_best:
            Path(args.write_best).write_text(str(best_path) + "\n")
        return

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
        cpu_eval_workers=args.cpu_eval_workers,
        cpu_eval_min_batch=args.cpu_eval_min_batch,
    )

    if args.parallel > 1 and env_config.cpu_eval_workers > 0 and not args.nested_eval_workers:
        print("note: disabling cpu_eval_workers for parallel selection (use --nested-eval-workers to keep).")
        env_config.cpu_eval_workers = 0

    payloads = [(path, env_config.__dict__, args) for path in paths]

    results = []
    if args.parallel > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.parallel) as pool:
            for path, score in pool.imap_unordered(_score_checkpoint, payloads):
                print(f"checkpoint={path} avg_return={score:.4f}", flush=True)
                results.append((path, score))
    else:
        for payload in payloads:
            path, score = _score_checkpoint(payload)
            print(f"checkpoint={path} avg_return={score:.4f}", flush=True)
            results.append((path, score))

    best_path, best_score = max(results, key=lambda item: item[1])
    print(f"best_checkpoint={best_path} best_avg_return={best_score:.4f}")
    if args.write_best:
        Path(args.write_best).write_text(best_path + "\n")


if __name__ == "__main__":
    main()
