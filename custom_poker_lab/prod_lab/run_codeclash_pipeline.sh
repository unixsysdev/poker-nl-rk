#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"

NUM_PLAYERS="${NUM_PLAYERS:-6}"
STACK="${STACK:-10000}"
SMALL_BLIND="${SMALL_BLIND:-5}"
BIG_BLIND="${BIG_BLIND:-10}"
DEVICE="${DEVICE:-cuda}"

BATCH_A="${BATCH_A:-512}"
ROLLOUT_A="${ROLLOUT_A:-4}"
HANDS_A="${HANDS_A:-20}"
EPISODES_A="${EPISODES_A:-400000}"

BATCH_B="${BATCH_B:-256}"
ROLLOUT_B="${ROLLOUT_B:-2}"
HANDS_B="${HANDS_B:-100}"
EPISODES_B="${EPISODES_B:-200000}"

BATCH_C="${BATCH_C:-128}"
ROLLOUT_C="${ROLLOUT_C:-1}"
HANDS_C="${HANDS_C:-1000}"
EPISODES_C="${EPISODES_C:-80000}"

PPO_EPOCHS="${PPO_EPOCHS:-8}"
MINIBATCH="${MINIBATCH:-8192}"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-4}"
CPU_EVAL_WORKERS="${CPU_EVAL_WORKERS:-4}"
CPU_EVAL_MIN_BATCH="${CPU_EVAL_MIN_BATCH:-64}"

LEAGUE_ROUNDS="${LEAGUE_ROUNDS:-4}"
LEAGUE_POP="${LEAGUE_POP:-6}"
LEAGUE_TOPK="${LEAGUE_TOPK:-2}"
LEAGUE_EPISODES="${LEAGUE_EPISODES:-100000}"
LEAGUE_ROLLOUT="${LEAGUE_ROLLOUT:-1}"

SAVE_A="${SAVE_A:-experiments/prod_nlhe_ppo_phaseA}"
SAVE_B="${SAVE_B:-experiments/prod_nlhe_ppo_phaseB}"
SAVE_C="${SAVE_C:-experiments/prod_nlhe_ppo_phaseC}"
SAVE_L="${SAVE_L:-experiments/prod_nlhe_league}"
SAVE_EVERY_A="${SAVE_EVERY_A:-20000}"
SAVE_EVERY_B="${SAVE_EVERY_B:-20000}"
SAVE_EVERY_C="${SAVE_EVERY_C:-20000}"
RESUME_A="${RESUME_A:-}"
RESUME_B="${RESUME_B:-}"
RESUME_C="${RESUME_C:-}"
RESUME_L="${RESUME_L:-}"

latest_ckpt() {
  local dir="$1"
  if [[ -d "$dir" ]]; then
    ls -1 "$dir"/policy_ep_*.pt 2>/dev/null | sort | tail -n 1
  fi
}

resolve_resume() {
  local explicit="$1"
  local primary="$2"
  local fallback="$3"
  if [[ -n "$explicit" ]]; then
    echo "$explicit"
    return
  fi
  local found
  found="$(latest_ckpt "$primary")"
  if [[ -n "$found" ]]; then
    echo "$found"
    return
  fi
  if [[ -n "$fallback" ]]; then
    found="$(latest_ckpt "$fallback")"
    if [[ -n "$found" ]]; then
      echo "$found"
      return
    fi
  fi
  echo ""
}

echo "phase_a: ppo warm-start"
RESUME_A_PATH="$(resolve_resume "$RESUME_A" "$SAVE_A" "")"
$PYTHON "$ROOT/custom_poker_lab/prod_lab/train_ppo.py" \
  --batch-size "$BATCH_A" --num-players "$NUM_PLAYERS" \
  --stack "$STACK" --small-blind "$SMALL_BLIND" --big-blind "$BIG_BLIND" \
  --hands-per-episode "$HANDS_A" \
  --episodes "$EPISODES_A" --rollout-episodes "$ROLLOUT_A" --device "$DEVICE" \
  --ppo-epochs "$PPO_EPOCHS" --minibatch "$MINIBATCH" --log-every-updates 1 --profile \
  --rollout-workers "$ROLLOUT_WORKERS" \
  --cpu-eval-workers "$CPU_EVAL_WORKERS" --cpu-eval-min-batch "$CPU_EVAL_MIN_BATCH" \
  --save-every "$SAVE_EVERY_A" --save-dir "$SAVE_A" \
  ${RESUME_A_PATH:+--resume "$RESUME_A_PATH"}

echo "phase_b: ppo mid-horizon"
RESUME_B_PATH="$(resolve_resume "$RESUME_B" "$SAVE_B" "$SAVE_A")"
$PYTHON "$ROOT/custom_poker_lab/prod_lab/train_ppo.py" \
  --batch-size "$BATCH_B" --num-players "$NUM_PLAYERS" \
  --stack "$STACK" --small-blind "$SMALL_BLIND" --big-blind "$BIG_BLIND" \
  --hands-per-episode "$HANDS_B" \
  --episodes "$EPISODES_B" --rollout-episodes "$ROLLOUT_B" --device "$DEVICE" \
  --ppo-epochs "$PPO_EPOCHS" --minibatch "$MINIBATCH" --log-every-updates 1 --profile \
  --rollout-workers "$ROLLOUT_WORKERS" \
  --cpu-eval-workers "$CPU_EVAL_WORKERS" --cpu-eval-min-batch "$CPU_EVAL_MIN_BATCH" \
  --save-every "$SAVE_EVERY_B" --save-dir "$SAVE_B" \
  ${RESUME_B_PATH:+--resume "$RESUME_B_PATH"}

echo "phase_c: ppo long-horizon (1000 hands)"
RESUME_C_PATH="$(resolve_resume "$RESUME_C" "$SAVE_C" "$SAVE_B")"
$PYTHON "$ROOT/custom_poker_lab/prod_lab/train_ppo.py" \
  --batch-size "$BATCH_C" --num-players "$NUM_PLAYERS" \
  --stack "$STACK" --small-blind "$SMALL_BLIND" --big-blind "$BIG_BLIND" \
  --hands-per-episode "$HANDS_C" \
  --episodes "$EPISODES_C" --rollout-episodes "$ROLLOUT_C" --device "$DEVICE" \
  --ppo-epochs "$PPO_EPOCHS" --minibatch "$MINIBATCH" --log-every-updates 1 --profile \
  --rollout-workers "$ROLLOUT_WORKERS" \
  --cpu-eval-workers "$CPU_EVAL_WORKERS" --cpu-eval-min-batch "$CPU_EVAL_MIN_BATCH" \
  --save-every "$SAVE_EVERY_C" --save-dir "$SAVE_C" \
  ${RESUME_C_PATH:+--resume "$RESUME_C_PATH"}

LATEST_PPO="$(latest_ckpt "$SAVE_C")"
if [[ -z "$LATEST_PPO" ]]; then
  LATEST_PPO="$(latest_ckpt "$SAVE_B")"
fi
if [[ -z "$LATEST_PPO" ]]; then
  LATEST_PPO="$(latest_ckpt "$SAVE_A")"
fi
echo "latest_ppo=${LATEST_PPO}"

echo "phase_d: league hardening"
RESUME_L_PATH="${RESUME_L:-$LATEST_PPO}"
$PYTHON "$ROOT/custom_poker_lab/prod_lab/train_league.py" \
  --batch-size "$BATCH_C" --num-players "$NUM_PLAYERS" \
  --stack "$STACK" --small-blind "$SMALL_BLIND" --big-blind "$BIG_BLIND" \
  --hands-per-episode "$HANDS_C" \
  --rounds "$LEAGUE_ROUNDS" --population "$LEAGUE_POP" --top-k "$LEAGUE_TOPK" \
  --episodes-per-agent "$LEAGUE_EPISODES" --rollout-episodes "$LEAGUE_ROLLOUT" --device "$DEVICE" \
  --ppo-epochs "$PPO_EPOCHS" --minibatch "$MINIBATCH" --log-every 1 --profile \
  --rollout-workers "$ROLLOUT_WORKERS" \
  --cpu-eval-workers "$CPU_EVAL_WORKERS" --cpu-eval-min-batch "$CPU_EVAL_MIN_BATCH" \
  --eval-episodes 2000 --eval-opponent proxy \
  --pool-size 8 --pool-prob 0.5 \
  --save-dir "$SAVE_L" \
  ${RESUME_L_PATH:+--resume "$RESUME_L_PATH"}

echo "league_complete: ${SAVE_L}"
