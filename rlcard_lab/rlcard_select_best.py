from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlcard_lab import rlcard_eval as evaler


CHECKPOINT_RE = re.compile(r"policy_ep_(\d+)\.pt")


def list_checkpoints(directory: pathlib.Path):
    items = []
    for path in directory.glob("policy_ep_*.pt"):
        match = CHECKPOINT_RE.match(path.name)
        if not match:
            continue
        items.append((int(match.group(1)), path))
    return [p for _, p in sorted(items, key=lambda x: x[0])]


def main():
    parser = argparse.ArgumentParser(description="Select best RLCard checkpoint.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--opponent", choices=["random", "policy", "lbr"], default="random")
    parser.add_argument("--opponent-policy", default=None)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lbr-rollouts", type=int, default=16)
    args = parser.parse_args()

    ckpt_dir = pathlib.Path(args.checkpoint_dir)
    checkpoints = list_checkpoints(ckpt_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")

    best = None
    for path in checkpoints:
        env = evaler.make_env(seed=args.seed, num_players=2, allow_step_back=args.opponent == "lbr")
        policy = evaler.load_policy(str(path), device=args.device)

        if args.opponent == "policy":
            if not args.opponent_policy:
                raise ValueError("--opponent-policy is required when --opponent=policy")
            opponent = evaler.load_policy(args.opponent_policy, device=args.device)
            score = evaler.evaluate(env, policy, opponent, args.episodes)
        elif args.opponent == "lbr":
            score = evaler.evaluate_lbr(env, policy, args.episodes, args.lbr_rollouts, args.seed)
        else:
            opponent = evaler.RandomPolicy(env.num_actions)
            score = evaler.evaluate(env, policy, opponent, args.episodes)

        print(f"checkpoint={path} avg_return={score:.4f}")
        if best is None or score > best[1]:
            best = (path, score)

    if best is not None:
        print(f"best_checkpoint={best[0]} best_avg_return={best[1]:.4f}")


if __name__ == "__main__":
    main()
