import argparse

import pyspiel
from open_spiel.python.algorithms import cfr, exploitability


def main() -> None:
    parser = argparse.ArgumentParser(description="CFR training on OpenSpiel poker.")
    parser.add_argument("--game", default="leduc_poker")
    parser.add_argument("--algorithm", choices=["cfr", "cfr_plus"], default="cfr")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=200)
    args = parser.parse_args()

    game = pyspiel.load_game(args.game)
    if args.algorithm == "cfr":
        solver = cfr.CFRSolver(game)
    else:
        solver = cfr.CFRPlusSolver(game)

    for i in range(1, args.iterations + 1):
        solver.evaluate_and_update_policy()
        if i % args.eval_every == 0 or i == args.iterations:
            policy = solver.average_policy()
            exp = exploitability.exploitability(game, policy)
            print(f"iter={i} exploitability={exp:.6f}")


if __name__ == "__main__":
    main()
