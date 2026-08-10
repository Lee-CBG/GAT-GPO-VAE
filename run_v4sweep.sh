#!/bin/bash
# GNN v4 pc-sweep — GPUs 1,2,3 only (GPU 0 = heavy tenant, excluded)
# Rules: max 1 K562 per GPU; skip-ahead backfill of small jobs into spare memory.
set -u
cd "$(dirname "$0")"
ROOT="$(pwd)"
mkdir -p logs/v4sweep

GPUS=(1 2 3)
declare -A BUDGET=( [1]=36000 [2]=42000 [3]=42000 )   # MiB free we may use
declare -A KCOUNT=( [1]=0 [2]=0 [3]=0 )               # running K562 per GPU
KMAX=1                                                  # hard cap: 1 K562/GPU

declare -A COST=(  [rpe1]=6000  [k562]=21000 [adamson]=8000 )
declare -A TRAIN=( [rpe1]=train_rpe1.py [k562]=train_replogle.py [adamson]=train_adamson.py )

# Queue: interleave so backfill is natural; K562 still placed where it fits.
JOBS=()
for f in demo/gnn_v4sweep_k562_*.yaml;    do JOBS+=("k562 $(basename "$f" .yaml)");    done
for f in demo/gnn_v4sweep_rpe1_*.yaml;    do JOBS+=("rpe1 $(basename "$f" .yaml)");    done
for f in demo/gnn_v4sweep_adamson_*.yaml; do JOBS+=("adamson $(basename "$f" .yaml)"); done
NJOBS=${#JOBS[@]}
declare -a DONE; for ((k=0;k<NJOBS;k++)); do DONE[$k]=0; done
echo "Total jobs: $NJOBS"

declare -A RUN_GPU RUN_COST RUN_DS

launch() {
  local ds=$1 cfg=$2 gpu=$3 cost=$4
  CUDA_VISIBLE_DEVICES=$gpu setsid python "$ROOT/${TRAIN[$ds]}" \
    --config "$ROOT/demo/${cfg}.yaml" \
    > "$ROOT/logs/v4sweep/${cfg}.log" 2>&1 &
  local pid=$!
  disown "$pid"
  RUN_GPU[$pid]=$gpu; RUN_COST[$pid]=$cost; RUN_DS[$pid]=$ds
  [ "$ds" = k562 ] && KCOUNT[$gpu]=$(( KCOUNT[$gpu]+1 ))
  echo "[launch] $cfg -> GPU $gpu (cost ${cost}, pid $pid)"
}

reap() {
  for pid in "${!RUN_GPU[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      local g=${RUN_GPU[$pid]} c=${RUN_COST[$pid]} d=${RUN_DS[$pid]}
      BUDGET[$g]=$(( BUDGET[$g]+c ))
      [ "$d" = k562 ] && KCOUNT[$g]=$(( KCOUNT[$g]-1 ))
      echo "[done]   $d pid $pid freed ${c} on GPU $g (budget ${BUDGET[$g]}, kcount ${KCOUNT[$g]})"
      unset RUN_GPU[$pid] RUN_COST[$pid] RUN_DS[$pid]
    fi
  done
}

# find a GPU for a given ds, honoring budget + K562 cap; prefer big cards 2,3 then 1
place() {
  local ds=$1 cost=$2 g
  for g in 2 3 1; do
    if [ "${BUDGET[$g]}" -ge "$cost" ]; then
      if [ "$ds" = k562 ] && [ "${KCOUNT[$g]}" -ge "$KMAX" ]; then continue; fi
      echo "$g"; return
    fi
  done
}

remaining() { local k; for ((k=0;k<NJOBS;k++)); do [ "${DONE[$k]}" -eq 0 ] && return 0; done; return 1; }

while remaining || [ "${#RUN_GPU[@]}" -gt 0 ]; do
  reap
  # skip-ahead: scan ALL undone jobs, launch any that fit
  progress=1
  while [ "$progress" -eq 1 ]; do
    progress=0
    for ((k=0;k<NJOBS;k++)); do
      [ "${DONE[$k]}" -eq 1 ] && continue
      entry=(${JOBS[$k]}); ds=${entry[0]}; cfg=${entry[1]}; cost=${COST[$ds]}
      gpu=$(place "$ds" "$cost")
      if [ -n "$gpu" ]; then
        BUDGET[$gpu]=$(( BUDGET[$gpu]-cost ))
        launch "$ds" "$cfg" "$gpu" "$cost"
        DONE[$k]=1; progress=1
      fi
    done
  done
  sleep 20
done
echo "ALL v4 SWEEP JOBS COMPLETE"
