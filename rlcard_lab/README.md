# RLCard NLHE Lab (Two-Head Policy)

This folder contains a lightweight RLCard setup for NLHE using a two-head
policy (action type + raise size). It is **discrete** and uses RLCard's
built-in no-limit hold'em action set:

- fold
- check/call
- raise half pot
- raise pot
- all-in

That keeps training tractable, but it is still an abstraction.

## Install

```bash
pip install -r requirements-rlcard.txt
```

## Train

```bash
python rlcard_lab/rlcard_train.py --episodes 50000 --update-every 200 \
  --log-every 1000 --eval-every 5000 --save-every 5000 \
  --save-dir experiments/rlcard_nlhe
```

### PPO + League (Recommended)

```bash
python rlcard_lab/rlcard_train_ppo.py --episodes 200000 \
  --rollout-episodes 200 --ppo-epochs 4 --minibatch 1024 \
  --log-every 5000 --eval-every 20000 --eval-episodes 2000 \
  --save-every 20000 --save-dir experiments/rlcard_nlhe_ppo \
  --pool-size 8 --pool-add-every 20000 --pool-prob 0.5
```

To resume from a checkpoint, pass `--resume` and set `--episodes` to your new
total target (e.g. resume from 50k up to 100k):

```bash
python rlcard_lab/rlcard_train.py --episodes 100000 --update-every 200 \
  --log-every 1000 --eval-every 5000 --save-every 5000 \
  --save-dir experiments/rlcard_nlhe \
  --resume experiments/rlcard_nlhe/policy_ep_050000.pt
```

To resume PPO training:

```bash
python rlcard_lab/rlcard_train_ppo.py --episodes 400000 \
  --rollout-episodes 200 --ppo-epochs 4 --minibatch 1024 \
  --log-every 5000 --eval-every 20000 --eval-episodes 2000 \
  --save-every 20000 --save-dir experiments/rlcard_nlhe_ppo \
  --pool-size 8 --pool-add-every 20000 --pool-prob 0.5 \
  --resume experiments/rlcard_nlhe_ppo/policy_ep_200000.pt
```

## Evaluate

```bash
python rlcard_lab/rlcard_eval.py --policy experiments/rlcard_nlhe/policy_ep_050000.pt \
  --opponent random --episodes 2000
```

Evaluate against a local best response (proxy):

```bash
python rlcard_lab/rlcard_eval.py --policy experiments/rlcard_nlhe/policy_ep_050000.pt \
  --opponent lbr --lbr-rollouts 16 --episodes 2000
```

To evaluate against another checkpoint:

```bash
python rlcard_lab/rlcard_eval.py --policy experiments/rlcard_nlhe/policy_ep_050000.pt \
  --opponent policy --opponent-policy experiments/rlcard_nlhe/policy_ep_025000.pt
```

Select the best checkpoint against LBR:

```bash
python rlcard_lab/rlcard_select_best.py --checkpoint-dir experiments/rlcard_nlhe \
  --opponent lbr --lbr-rollouts 16 --episodes 2000
```
