# CUDA NLHE (GPU-Native Env)

This folder contains a **torch-native** vectorized NLHE environment. The
environment state and step logic live on CUDA, which allows rollouts to run
largely on GPU. Showdown evaluation still uses **Treys** on CPU, so this is not
100% GPU end-to-end yet (hand evaluation is the remaining CPU hot spot).

## Train (CUDA PPO)

```bash
python custom_poker_lab/cuda_lab/train_ppo.py --batch-size 256 --num-players 6 \
  --episodes 400000 --rollout-episodes 8 --device cuda \
  --ppo-epochs 8 --minibatch 8192 \
  --save-every 100000 --save-dir experiments/cuda_nlhe_ppo
```

## Evaluate

```bash
python custom_poker_lab/cuda_lab/eval.py \
  --policy experiments/cuda_nlhe_ppo/policy_ep_100000.pt \
  --episodes 2000 --device cuda
```

Notes:
- Continuous raise sizing: bet fraction scales **from min-raise to all-in**.
- Hand evaluation is still CPU (Treys). Replace with a GPU evaluator for full
  end-to-end CUDA rollouts.
