# Prod NLHE (Vectorized Engine)

This folder contains a **vectorized, rules-complete NLHE engine** aimed at
high-throughput training and evaluation. It keeps the original `custom_poker_lab`
implementation intact and provides a faster, batch-oriented environment.

Highlights:
- Vectorized environment (`batch_size` parallel tables)
- Full NLHE rules: blinds, antes, min-raise logic, side pots, split pots
- Rake caps per hand/per street
- Multi-hand episodes
- Action-history window in observations

It is still a research engine (not casino-certified), but the rules are complete
for NLHE experiments.

## Train (Vectorized PPO)

```bash
python custom_poker_lab/prod_lab/train_ppo.py --batch-size 64 --num-players 6 \
  --episodes 200000 --rollout-episodes 4 --device cuda \
  --log-every 10000 --save-every 50000 \
  --save-dir experiments/prod_nlhe_ppo
```

## Evaluate

```bash
python custom_poker_lab/prod_lab/eval.py \
  --policy experiments/prod_nlhe_ppo/policy_ep_050000.pt \
  --opponent random --episodes 2000
```

LBR proxy:

```bash
python custom_poker_lab/prod_lab/eval.py \
  --policy experiments/prod_nlhe_ppo/policy_ep_050000.pt \
  --opponent lbr --lbr-rollouts 32 --lbr-bet-fracs 0.25,0.5,1.0 \
  --episodes 2000
```

Depth-limited best response (slower, stronger proxy):

```bash
python custom_poker_lab/prod_lab/eval.py \
  --policy experiments/prod_nlhe_ppo/policy_ep_050000.pt \
  --opponent dlbr --br-depth 2 --br-other-samples 1 \
  --lbr-rollouts 16 --lbr-bet-fracs 0.25,0.5,1.0 \
  --episodes 500
```

## League Training (Population)

```bash
python custom_poker_lab/prod_lab/train_league.py --batch-size 64 --num-players 6 \
  --rounds 4 --population 6 --top-k 2 \
  --episodes-per-agent 20000 --rollout-episodes 4 --device cuda \
  --eval-episodes 2000 --eval-opponent proxy \
  --pool-size 8 --pool-prob 0.5 \
  --save-dir experiments/prod_nlhe_league
```
