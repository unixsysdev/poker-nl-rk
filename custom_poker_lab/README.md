# Custom NLHE (Multi-Player, Parameterized Bet Sizing)

This is a research-only environment that supports **multi-player NLHE** with
parameterized bet sizes (continuous fraction of pot). It uses a two-head policy:

- **Action type:** fold / call-check / raise
- **Bet fraction:** continuous in [0, 1], scaled to [0, 2]x pot, clipped to min raise

Notes:
- Supports multi-player hands (2+ players).
- Uses side-pot resolution, rake caps (per hand/per street), ante, and multi-hand episodes.
- Observation includes per-player features plus an action-history window.
- Still a research env (rules-complete for our experiments), not a production/casino-certified engine.

## Install

```bash
pip install -r requirements-custom-poker.txt
```

## Train (PPO)

```bash
python custom_poker_lab/poker_train_ppo.py --num-players 4 \
  --episodes 200000 --rollout-episodes 200 --workers 4 \
  --ppo-epochs 4 --minibatch 1024 \
  --log-every 5000 --save-every 20000 \
  --save-dir experiments/custom_nlhe_ppo
```

You can also tune the game config:

```bash
python custom_poker_lab/poker_train_ppo.py --num-players 6 \
  --stack 20000 --small-blind 50 --big-blind 100 \
  --max-raises 4 --ante 0 --rake-pct 0.0 --rake-cap 0 \
  --rake-cap-hand 0 --rake-cap-street 0 \
  --history-len 12 --hands-per-episode 1 \
  --episodes 200000 --rollout-episodes 200 --workers 6 \
  --ppo-epochs 4 --minibatch 1024 \
  --log-every 5000 --save-every 20000 \
  --save-dir experiments/custom_nlhe_ppo
```

You can enable periodic evaluation (random or LBR proxy):

```bash
python custom_poker_lab/poker_train_ppo.py --num-players 4 \
  --episodes 200000 --rollout-episodes 200 --workers 4 \
  --ppo-epochs 4 --minibatch 1024 \
  --eval-every 20000 --eval-episodes 2000 --eval-opponent lbr \
  --lbr-rollouts 32 --lbr-bet-fracs 0.5,1.0,2.0 \
  --log-every 5000 --save-every 20000 \
  --save-dir experiments/custom_nlhe_ppo
```

## Evaluate

```bash
python custom_poker_lab/poker_eval.py \
  --policy experiments/custom_nlhe_ppo/policy_ep_200000.pt \
  --num-players 4 --episodes 2000
```

LBR proxy eval (parallel):

```bash
python custom_poker_lab/poker_eval.py \
  --policy experiments/custom_nlhe_ppo/policy_ep_200000.pt \
  --opponent lbr --lbr-rollouts 32 --lbr-bet-fracs 0.5,1.0,2.0 \
  --episodes 2000 --eval-parallel 4
```

## League Training (Population)

```bash
python custom_poker_lab/poker_train_league.py --num-players 4 \
  --rounds 4 --population 6 --top-k 2 \
  --episodes-per-agent 20000 --rollout-episodes 200 --workers 4 \
  --eval-episodes 2000 --eval-opponent lbr --eval-parallel 4 \
  --pool-size 8 --pool-prob 0.5 \
  --save-dir experiments/custom_nlhe_league
```
