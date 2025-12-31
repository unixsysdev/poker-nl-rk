# Poker RL (Research Only)

This repo scaffolds poker RL experiments in simulation (OpenSpiel). It is not
for real-money play or bypassing site rules.

## Quick Start (Leduc)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_cfr.py --game leduc_poker --iterations 2000 --eval-every 200
```

## OpenSpiel Poker Env (Gymnasium)

The Gymnasium wrapper lives in `poker_rl/envs/openspiel_poker_env.py` and gives
you a single-agent view with an internal opponent. This makes it easy to plug
into RL code and verify correctness on Leduc before scaling.

Example:

```python
from poker_rl.envs import OpenSpielPokerEnv
from poker_rl.solvers import CFRPolicy, SolverAuditPolicy

env = OpenSpielPokerEnv("leduc_poker")
opponent = CFRPolicy("leduc_poker", iterations=2000)
env.set_opponent_policy(opponent)
```

## Test-Time "Solver" Augmentation

You can use `CFRPolicy` as a strong opponent or as a test-time check. For small
games (Leduc), CFR can get very close to optimal quickly, which is useful to
validate your learning code.

For larger games (HUNL/NLHE), use abstractions and fewer CFR iterations, or do
depth-limited re-solving on subgames rather than full brute force.

Example audit wrapper (gates a policy with a CFR check):

```python
from poker_rl.solvers import CFRPolicy, SolverAuditPolicy

solver = CFRPolicy("leduc_poker", iterations=2000)
auditor = SolverAuditPolicy(
    base_policy=lambda state, rng: rng.choice(state.legal_actions(state.current_player())),
    solver_policy=solver,
    min_solver_prob=0.05,
    fallback="argmax",
)
```

## Alpha-Style Population (No PufferLib)

If you want to avoid PufferLib entirely, use the population trainer that runs
directly on OpenSpiel with a lightweight policy/value network:

```bash
python league_train_alpha.py --game leduc_poker --rounds 5 --population 4 --top-k 2
```

This is a simple Alpha-style loop: train a population against a hall of fame,
score round-robin (and optionally vs CFR), then keep the top-K for the next round.

To get progress feedback during training, use:

```bash
python league_train_alpha.py --game leduc_poker --rounds 5 --population 4 --top-k 2 --log-every 500
```

To batch multiple episodes per update (better GPU utilization):

```bash
python league_train_alpha.py --game leduc_poker --batch-episodes 8
```

To train multiple candidates in parallel (CPU by default):

```bash
python league_train_alpha.py --game leduc_poker --parallel-agents 4 --worker-device cpu
```

To parallelize evaluation (round-robin + CFR) across processes:

```bash
python league_train_alpha.py --game leduc_poker --eval-parallel 4
```

## Evaluate a Trained Policy

The league run saves checkpoints to `experiments/alpha_league/round_XX.pt`.
Evaluate vs CFR or random:

```bash
python eval_alpha.py --model experiments/alpha_league/round_01.pt --opponent cfr --episodes 500
python eval_alpha.py --model experiments/alpha_league/round_01.pt --opponent random --episodes 500
```

To resume from a saved checkpoint:

```bash
python league_train_alpha.py --game leduc_poker --resume-model experiments/alpha_league/round_01.pt
```

By default, the trainer will auto-resume from the latest checkpoint in
`--output-dir` and continue round numbering. Use `--no-auto-resume` to disable it.

## Useful Commands

List available OpenSpiel games:

```bash
python - <<'PY'
import pyspiel
print(sorted(pyspiel.registered_names()))
PY
```

## Universal Poker (Limit Hold'em via Deep CFR)

Your OpenSpiel build exposes `universal_poker`, which can be configured to
approximate 2-player Limit Hold'em. The default config uses a **reduced deck**
(`num_ranks=6`) to keep training tractable.

Train Deep CFR (LHE-like):

```bash
python train_universal_deep_cfr.py --iterations 200 --traversals 200 --num-ranks 6
```

To move closer to real LHE, increase the deck size:

```bash
python train_universal_deep_cfr.py --iterations 200 --traversals 200 --num-ranks 13
```

If you see `invalid first player` errors, ensure your `--first-player` string
uses 1-based player IDs (e.g., `1 1 1 1` for four rounds).

If `betting="limit"` is not supported in your build, change it to `nolimit`
inside `poker_rl/universal_poker.py` and use `bettingAbstraction="fcpa"`.

If `deep_cfr` is not available in your OpenSpiel build, try:

```bash
python train_universal_deep_cfr.py --algorithm outcome_sampling_mccfr --iterations 5000
```

### Best-Approx NLHE (Single Run)

Use the preset to target a stronger NLHE-like configuration:

```bash
python train_universal_deep_cfr.py --preset best_nlhe \
  --algorithm outcome_sampling_mccfr \
  --iterations 5000000 --log-every 100000 --eval-every 500000 --eval-episodes 5000 \
  --checkpoint-every 500000 --checkpoint-dir experiments/universal_poker_mccfr_best
```

For a closer-to-real stack depth and raise sizing, try:

```bash
python train_universal_deep_cfr.py --preset real_nlhe \
  --algorithm outcome_sampling_mccfr \
  --iterations 5000000 --log-every 100000 --eval-every 500000 --eval-episodes 5000 \
  --checkpoint-every 500000 --checkpoint-dir experiments/universal_poker_mccfr_real
```

For a more granular (but slower) NLHE-like abstraction:

```bash
python train_universal_deep_cfr.py --preset granular_nlhe \
  --algorithm outcome_sampling_mccfr \
  --iterations 5000000 --log-every 100000 --eval-every 500000 --eval-episodes 5000 \
  --checkpoint-every 500000 --checkpoint-dir experiments/universal_poker_mccfr_granular
```

Note: `granular_nlhe` increases raise steps per round and uses tighter sizing,
but still relies on the `fcpa` action abstraction. Going beyond this requires
custom abstractions or a different engine.

Add progress logging, eval, and checkpoints:

```bash
python train_universal_deep_cfr.py --algorithm outcome_sampling_mccfr \
  --iterations 20000 --log-every 1000 --eval-every 5000 --eval-episodes 200 \
  --checkpoint-every 5000 --checkpoint-dir experiments/universal_poker_mccfr
```

Evaluate a saved universal_poker policy:

```bash
python eval_universal_policy.py --policy experiments/universal_poker_mccfr/policy_iter_100000.pkl \
  --config nlhe --num-ranks 13 --betting-abstraction fcpa --opponent random --episodes 2000
```

Evaluate vs a CFR baseline (slow for large games):

```bash
python eval_universal_policy.py --policy experiments/universal_poker_mccfr/policy_iter_100000.pkl \
  --config nlhe --num-ranks 13 --betting-abstraction fcpa --opponent cfr \
  --cfr-iterations 200 --episodes 500
```

## RLCard NLHE (Two-Head Policy)

If you want a separate RLCard-based setup with a two-head policy (action type +
raise size), see `rlcard_lab/README.md`. This uses RLCard's built-in NLHE action
set (fold / check-call / half-pot / pot / all-in).

## Custom NLHE (Multi-Player + Continuous Bet Sizing)

For a research-only multi-player NLHE environment with parameterized bet sizes,
see `custom_poker_lab/README.md`.

Evaluate vs a local best-response proxy (LBR):

```bash
python eval_universal_policy.py --policy experiments/universal_poker_mccfr/policy_iter_100000.pkl \
  --config nlhe --num-ranks 13 --betting-abstraction fcpa --opponent lbr \
  --lbr-rollouts 32 --episodes 2000
```

### Parallel Multi-Seed MCCFR (Best-Approx NLHE)

Run multiple seeds in parallel and write a summary:

```bash
python train_universal_mccfr_multi.py --preset best_nlhe \
  --seeds 41,42,43,44 --parallel 4 \
  --algorithm outcome_sampling_mccfr \
  --iterations 5000000 --log-every 100000 --eval-every 500000 --eval-episodes 5000 \
  --checkpoint-every 500000 --checkpoint-dir experiments/universal_poker_mccfr_multi
```

If you want to see live logs in the console instead of per-seed log files:

```bash
python train_universal_mccfr_multi.py --preset best_nlhe \
  --seeds 41,42,43,44 --parallel 4 \
  --algorithm outcome_sampling_mccfr \
  --iterations 5000000 --log-every 100000 --eval-every 500000 --eval-episodes 5000 \
  --checkpoint-every 500000 --checkpoint-dir experiments/universal_poker_mccfr_multi \
  --no-log-redirect
```

### Select the Best Seed (Proxy Eval)

```bash
python select_best_mccfr_seed.py --root-dir experiments/universal_poker_mccfr_multi \
  --opponent lbr --lbr-rollouts 32 --episodes 2000 --parallel 4
```
