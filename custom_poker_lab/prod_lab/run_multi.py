from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _with_seeded_save_dir(args: list[str], seed: int, mode: str) -> list[str]:
    updated = list(args)
    default_dir = f"experiments/prod_nlhe_{mode}_seed{seed}"

    for idx, arg in enumerate(updated):
        if arg == "--save-dir" and idx + 1 < len(updated):
            updated[idx + 1] = f"{updated[idx + 1]}_seed{seed}"
            return updated
        if arg.startswith("--save-dir="):
            base = arg.split("=", 1)[1]
            updated[idx] = f"--save-dir={base}_seed{seed}"
            return updated

    updated.extend(["--save-dir", default_dir])
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple prod_lab trainings in parallel.")
    parser.add_argument("--mode", choices=["ppo", "league"], required=True)
    parser.add_argument("--seeds", required=True, help="Comma-separated seeds.")
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("No seeds provided.")

    if args.args and args.args[0] == "--":
        extra_args = args.args[1:]
    else:
        extra_args = args.args

    script = "custom_poker_lab/prod_lab/train_ppo.py" if args.mode == "ppo" else "custom_poker_lab/prod_lab/train_league.py"
    script_path = str(Path(__file__).resolve().parents[2] / script)

    running: list[subprocess.Popen] = []
    for seed in seeds:
        cmd_args = _with_seeded_save_dir(extra_args, seed, args.mode)
        cmd = [sys.executable, script_path, "--seed", str(seed), *cmd_args]
        print("launch:", " ".join(cmd), flush=True)
        while len(running) >= max(1, args.parallel):
            proc = running.pop(0)
            proc.wait()
        running.append(subprocess.Popen(cmd))

    for proc in running:
        proc.wait()


if __name__ == "__main__":
    main()
