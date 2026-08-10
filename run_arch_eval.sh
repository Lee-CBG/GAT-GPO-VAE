#!/bin/bash
# Eval all 72 arch-sweep checkpoints. ONE eval per GPU (peaks ~28-43GB). Pre-resolved paths.
set -u
cd "$(dirname "$0")"; ROOT="$(pwd)"; mkdir -p logs/arch_eval
canon () {
  local cfg="$1" best="" bestep=-1 f ep d
  while read f; do [ -z "$f" ] && continue
    ep=$(echo "$f"|grep -oP 'epoch=\K[0-9]+')
    [ "$ep" -gt "$bestep" ] && { bestep=$ep; d=$(echo "$f"|sed 's|/checkpoints/.*||'); best=$d; }
  done < <(find results -path "*${cfg}*/checkpoints/best*.ckpt" 2>/dev/null | grep -E "/${cfg}(-[0-9]+)?/checkpoints/")
  echo "$best"
}
PEND=()
for ds in k562 rpe1 adamson; do
  for tag in L1 L3 H2 H8 D128 D512 drop0p1 drop0p3; do
    for s in 0 1 2; do
      cfg=gnn_arch_${ds}_${tag}_seed${s}
      d=$(canon "$cfg"); [ -z "$d" ] && { echo "NO DIR: $cfg"; continue; }
      [ -f "$d/test_metrics.csv" ] && continue
      PEND+=("$cfg|$d")
    done
  done
done
NP=${#PEND[@]}; echo "Pending arch evals: $NP"
GPUS=(1 2 3); declare -A BUSY=( [1]="" [2]="" [3]="" ); i=0
while [ "$i" -lt "$NP" ] || [ -n "${BUSY[1]}${BUSY[2]}${BUSY[3]}" ]; do
  for g in "${GPUS[@]}"; do
    if [ -n "${BUSY[$g]}" ] && ! kill -0 "${BUSY[$g]}" 2>/dev/null; then echo "[done] GPU $g"; BUSY[$g]=""; fi
  done
  for g in "${GPUS[@]}"; do
    if [ -z "${BUSY[$g]}" ] && [ "$i" -lt "$NP" ]; then
      entry="${PEND[$i]}"; cfg="${entry%%|*}"; dir="${entry##*|}"
      CUDA_VISIBLE_DEVICES=$g WANDB_MODE=disabled setsid python "$ROOT/eval.py" \
        --experiment_path "$dir" --perturbseq --batch_size 128 --ate_n_particles 2500 \
        --devices 0 --thr 3 > "$ROOT/logs/arch_eval/${cfg}.log" 2>&1 &
      BUSY[$g]=$!; echo "[launch] $cfg -> GPU $g ($dir)"; i=$((i+1))
    fi
  done
  sleep 15
done
echo "ALL ARCH EVALS COMPLETE"
