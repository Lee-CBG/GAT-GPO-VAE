#!/bin/bash
set -u
cd "$(dirname "$0")"; ROOT="$(pwd)"; mkdir -p logs/topup
declare -A BUDGET=( [1]=36000 [2]=42000 [3]=42000 )
declare -A KCOUNT=( [1]=0 [2]=0 [3]=0 ); KMAX=1
declare -A COST=( [rpe1]=6000 [k562]=21000 [adamson]=8000 )
declare -A TRAIN=( [rpe1]=train_rpe1.py [k562]=train_replogle.py [adamson]=train_adamson.py )

JOBS=()
for cfg in gnn_v4sweep_rpe1_pc12_seed3 gnn_v4sweep_rpe1_pc12_seed4 \
           gnn_v4sweep_rpe1_pc18_seed3 gnn_v4sweep_rpe1_pc18_seed4 \
           gnn_v4sweep_k562_pc4_seed3 gnn_v4sweep_k562_pc4_seed4 \
           gnn_v4sweep_k562_pc7_seed3 gnn_v4sweep_k562_pc7_seed4 \
           gnn_v4sweep_adamson_pc0p7_seed3 gnn_v4sweep_adamson_pc0p7_seed4 \
           gnn_v4sweep_adamson_pc1_seed3 gnn_v4sweep_adamson_pc1_seed4; do
  if   [[ $cfg == *_rpe1_*    ]]; then ds=rpe1
  elif [[ $cfg == *_k562_*    ]]; then ds=k562
  else ds=adamson; fi
  JOBS+=("$ds $cfg")
done
NJOBS=${#JOBS[@]}; declare -a DONE; for ((k=0;k<NJOBS;k++)); do DONE[$k]=0; done
echo "Top-up training jobs: $NJOBS"
declare -A RUN_GPU RUN_COST RUN_DS
launch(){ local ds=$1 cfg=$2 gpu=$3 cost=$4
  CUDA_VISIBLE_DEVICES=$gpu setsid python "$ROOT/${TRAIN[$ds]}" --config "$ROOT/demo/${cfg}.yaml" \
    > "$ROOT/logs/topup/${cfg}.log" 2>&1 & local pid=$!; disown $pid
  RUN_GPU[$pid]=$gpu; RUN_COST[$pid]=$cost; RUN_DS[$pid]=$ds
  [ "$ds" = k562 ] && KCOUNT[$gpu]=$((KCOUNT[$gpu]+1)); echo "[launch] $cfg -> GPU $gpu (pid $pid)"; }
reap(){ for pid in "${!RUN_GPU[@]}"; do if ! kill -0 $pid 2>/dev/null; then
  local g=${RUN_GPU[$pid]} c=${RUN_COST[$pid]} d=${RUN_DS[$pid]}
  BUDGET[$g]=$((BUDGET[$g]+c)); [ "$d" = k562 ] && KCOUNT[$g]=$((KCOUNT[$g]-1))
  echo "[done] $d pid $pid (GPU $g)"; unset RUN_GPU[$pid] RUN_COST[$pid] RUN_DS[$pid]; fi; done; }
place(){ local ds=$1 cost=$2 g; for g in 2 3 1; do if [ "${BUDGET[$g]}" -ge "$cost" ]; then
  [ "$ds" = k562 ] && [ "${KCOUNT[$g]}" -ge "$KMAX" ] && continue; echo "$g"; return; fi; done; }
rem(){ local k; for ((k=0;k<NJOBS;k++)); do [ "${DONE[$k]}" -eq 0 ] && return 0; done; return 1; }
while rem || [ "${#RUN_GPU[@]}" -gt 0 ]; do reap; p=1
  while [ "$p" -eq 1 ]; do p=0; for ((k=0;k<NJOBS;k++)); do [ "${DONE[$k]}" -eq 1 ] && continue
    e=(${JOBS[$k]}); ds=${e[0]}; cfg=${e[1]}; cost=${COST[$ds]}; gpu=$(place "$ds" "$cost")
    [ -z "$gpu" ] && continue; BUDGET[$gpu]=$((BUDGET[$gpu]-cost)); launch "$ds" "$cfg" "$gpu" "$cost"
    DONE[$k]=1; p=1; done; done; sleep 20; done
echo "ALL TOPUP TRAINING COMPLETE"
