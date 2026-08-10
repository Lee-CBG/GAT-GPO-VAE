#!/bin/bash
# Eval all 39 v4-sweep configs under corrected pipeline (FC3 + test-split).
# Resolves canonical (highest-epoch) dir per config; K562 eval ~43GB => 1/GPU.
set -u
cd "$(dirname "$0")"
ROOT="$(pwd)"
mkdir -p logs/v4sweep_eval

GPUS=(1 2 3)
# Eval budgets (MiB free we may use). K562 eval ~43GB -> only fits on big empty cards.
declare -A BUDGET=( [1]=36000 [2]=44000 [3]=44000 )
declare -A COST=(  [rpe1]=10000 [k562]=44000 [adamson]=8000 )   # eval footprints
declare -A KCOUNT=( [1]=0 [2]=0 [3]=0 ); KMAX=1                  # 1 K562 eval/GPU

# Resolve canonical dir (highest epoch) for a config name
canon_dir() {
  local cfg=$1 best="" bestep=-1 f ep d
  while read f; do
    [ -z "$f" ] && continue
    ep=$(echo "$f" | grep -oP 'epoch=\K[0-9]+')
    if [ "$ep" -gt "$bestep" ]; then bestep=$ep; d=$(echo "$f" | sed 's|/checkpoints/.*||'); best=$d; fi
  done < <(find results -path "*${cfg}*/checkpoints/best*.ckpt" 2>/dev/null | grep -E "/${cfg}(-[0-9]+)?/checkpoints/")
  echo "$best"
}

# Build job list
JOBS=()
for ds in k562 rpe1 adamson; do
  for f in demo/gnn_v4sweep_${ds}_*.yaml; do
    cfg=$(basename "$f" .yaml)
    d=$(canon_dir "$cfg")
    [ -n "$d" ] && JOBS+=("$ds $cfg $d")
  done
done
NJOBS=${#JOBS[@]}
declare -a DONE; for ((k=0;k<NJOBS;k++)); do DONE[$k]=0; done
echo "Eval jobs: $NJOBS"

declare -A RUN_GPU RUN_COST RUN_DS
launch() {
  local ds=$1 cfg=$2 dir=$3 gpu=$4 cost=$5
  CUDA_VISIBLE_DEVICES=$gpu WANDB_MODE=disabled setsid python "$ROOT/eval.py" \
    --experiment_path "$dir" --perturbseq --batch_size 128 \
    --ate_n_particles 2500 --devices 0 --thr 3 \
    > "$ROOT/logs/v4sweep_eval/${cfg}.log" 2>&1 &
  local pid=$!; disown "$pid"
  RUN_GPU[$pid]=$gpu; RUN_COST[$pid]=$cost; RUN_DS[$pid]=$ds
  [ "$ds" = k562 ] && KCOUNT[$gpu]=$(( KCOUNT[$gpu]+1 ))
  echo "[eval] $cfg -> GPU $gpu  ($dir)"
}
reap() {
  for pid in "${!RUN_GPU[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      local g=${RUN_GPU[$pid]} c=${RUN_COST[$pid]} d=${RUN_DS[$pid]}
      BUDGET[$g]=$(( BUDGET[$g]+c )); [ "$d" = k562 ] && KCOUNT[$g]=$(( KCOUNT[$g]-1 ))
      echo "[done] $d pid $pid (GPU $g, budget ${BUDGET[$g]})"
      unset RUN_GPU[$pid] RUN_COST[$pid] RUN_DS[$pid]
    fi
  done
}
place() {
  local ds=$1 cost=$2 g
  for g in 2 3 1; do
    if [ "${BUDGET[$g]}" -ge "$cost" ]; then
      [ "$ds" = k562 ] && [ "${KCOUNT[$g]}" -ge "$KMAX" ] && continue
      echo "$g"; return
    fi
  done
}
remaining() { local k; for ((k=0;k<NJOBS;k++)); do [ "${DONE[$k]}" -eq 0 ] && return 0; done; return 1; }

while remaining || [ "${#RUN_GPU[@]}" -gt 0 ]; do
  reap
  prog=1
  while [ "$prog" -eq 1 ]; do
    prog=0
    for ((k=0;k<NJOBS;k++)); do
      [ "${DONE[$k]}" -eq 1 ] && continue
      e=(${JOBS[$k]}); ds=${e[0]}; cfg=${e[1]}; dir=${e[2]}; cost=${COST[$ds]}
      gpu=$(place "$ds" "$cost")
      [ -z "$gpu" ] && continue
      BUDGET[$gpu]=$(( BUDGET[$gpu]-cost )); launch "$ds" "$cfg" "$dir" "$gpu" "$cost"
      DONE[$k]=1; prog=1
    done
  done
  sleep 20
done
echo "ALL v4 SWEEP EVALS COMPLETE"
