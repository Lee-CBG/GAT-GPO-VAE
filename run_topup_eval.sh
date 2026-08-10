#!/bin/bash
# Eval all remaining seed-3,4 checkpoints. ONE eval per GPU (evals peak ~28-43GB).
# Paths pre-resolved into an explicit list to avoid variable races.
set -u
cd "$(dirname "$0")"; ROOT="$(pwd)"; mkdir -p logs/topup_eval

canon () {
  local cfg="$1" best="" bestep=-1 f ep d
  while read f; do [ -z "$f" ] && continue
    ep=$(echo "$f"|grep -oP 'epoch=\K[0-9]+')
    [ "$ep" -gt "$bestep" ] && { bestep=$ep; d=$(echo "$f"|sed 's|/checkpoints/.*||'); best=$d; }
  done < <(find results -path "*${cfg}*/checkpoints/best*.ckpt" 2>/dev/null | grep -E "/${cfg}(-[0-9]+)?/checkpoints/")
  echo "$best"
}

# All 18 seed-3,4 configs. Skip any that already has test_metrics.csv.
ALL=(
  gnn_v4sweep_rpe1_pc12_seed3 gnn_v4sweep_rpe1_pc12_seed4
  gnn_v4sweep_rpe1_pc18_seed3 gnn_v4sweep_rpe1_pc18_seed4
  gnn_v4_rpe1_pc15_seed3 gnn_v4_rpe1_pc15_seed4
  gnn_v4sweep_k562_pc4_seed3 gnn_v4sweep_k562_pc4_seed4
  gnn_v4sweep_k562_pc7_seed3 gnn_v4sweep_k562_pc7_seed4
  gnn_v4_k562_pc5_seed3 gnn_v4_k562_pc5_seed4
  gnn_v4sweep_adamson_pc0p7_seed3 gnn_v4sweep_adamson_pc0p7_seed4
  gnn_v4sweep_adamson_pc1_seed3 gnn_v4sweep_adamson_pc1_seed4
  gnn_v4_adamson_pc0p5_seed3 gnn_v4_adamson_pc0p5_seed4
)

# Build pending list: "cfg|dir", skip if metrics already exist
PEND=()
for cfg in "${ALL[@]}"; do
  d=$(canon "$cfg")
  [ -z "$d" ] && { echo "NO DIR: $cfg"; continue; }
  if [ -f "$d/test_metrics.csv" ]; then echo "skip (done): $cfg"; continue; fi
  PEND+=("$cfg|$d")
done
NP=${#PEND[@]}
echo "Pending evals: $NP"

GPUS=(1 2 3)
declare -A BUSY=( [1]="" [2]="" [3]="" )   # pid running on each gpu
i=0
while [ "$i" -lt "$NP" ] || [ -n "${BUSY[1]}${BUSY[2]}${BUSY[3]}" ]; do
  # reap finished
  for g in "${GPUS[@]}"; do
    if [ -n "${BUSY[$g]}" ] && ! kill -0 "${BUSY[$g]}" 2>/dev/null; then
      echo "[done] GPU $g (pid ${BUSY[$g]})"; BUSY[$g]=""
    fi
  done
  # assign next pending to a free gpu
  for g in "${GPUS[@]}"; do
    if [ -z "${BUSY[$g]}" ] && [ "$i" -lt "$NP" ]; then
      entry="${PEND[$i]}"; cfg="${entry%%|*}"; dir="${entry##*|}"
      CUDA_VISIBLE_DEVICES=$g WANDB_MODE=disabled setsid python "$ROOT/eval.py" \
        --experiment_path "$dir" --perturbseq --batch_size 128 --ate_n_particles 2500 \
        --devices 0 --thr 3 > "$ROOT/logs/topup_eval/${cfg}.log" 2>&1 &
      BUSY[$g]=$!
      echo "[launch] $cfg -> GPU $g ($dir) pid ${BUSY[$g]}"
      i=$((i+1))
    fi
  done
  sleep 15
done
echo "ALL TOPUP EVALS COMPLETE"
