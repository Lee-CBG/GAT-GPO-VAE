#!/bin/bash
set -u
GUIDE=gpo_vae/models/gpo_vae/guides/gnn_guide.py
V3=gpo_vae/models/gpo_vae/guides/gnn_guide.py.v3
V4=gpo_vae/models/gpo_vae/guides/gnn_guide.py.v4

############ PHASE 1: TRAIN 3 K562 seeds, one per GPU, in parallel ############
cp "$V3" "$GUIDE"
grep -q "v3 no-detach ACTIVE" "$GUIDE" || { echo "ABORT: guide not v3"; exit 1; }
echo "=== PHASE 1: K562 v3 training (guide=v3) $(date +%T) ==="

train_one () {  # gpu seed
  echo "[train GPU$1] k562 seed$2 START $(date +%T)"
  CUDA_VISIBLE_DEVICES=$1 python train_replogle.py --config ./demo/gnn_v3_k562_pc5_seed${2}.yaml \
    > logs/train_gnn_v3_k562_pc5_seed${2}_final.log 2>&1
  echo "[train GPU$1] k562 seed$2 DONE  $(date +%T)"
}

train_one 1 0 &
train_one 2 2 &
train_one 3 3 &
wait
echo "=== PHASE 1 COMPLETE: all 3 K562 seeds trained $(date +%T) ==="

# restore v4 (training done)
cp "$V4" "$GUIDE"
echo "guide restored -> v4 $(date +%T)"

############ PHASE 2: EVAL — per dataset per GPU, serial within dataset ############
echo "=== PHASE 2: eval all 15 v3 (FC3 + test-split) $(date +%T) ==="

eval_dataset () {  # gpu  name1 name2 ...
  local gpu=$1; shift
  for name in "$@"; do
    d=$(find results/${name}* -maxdepth 0 -type d 2>/dev/null | grep -vE "FAILED|BAD|partial" | head -1)
    if [ -z "$d" ] || [ -z "$(find "$d/checkpoints" -name 'best*.ckpt' 2>/dev/null)" ]; then
      echo "[eval GPU$gpu] $name MISSING — skip"; continue
    fi
    echo "[eval GPU$gpu] $name START $(date +%T)"
    CUDA_VISIBLE_DEVICES=$gpu WANDB_MODE=disabled python eval.py \
      --experiment_path "$d" --perturbseq --batch_size 128 \
      --ate_n_particles 2500 --devices 0 --thr 3 \
      > logs/eval_${name}.log 2>&1
    echo "[eval GPU$gpu] $name DONE  $(date +%T)"
  done
}

# RPE1 -> GPU 1 ; K562 -> GPU 2 ; Adamson -> GPU 3 (datasets parallel, seeds serial)
eval_dataset 1 gnn_v3_rpe1_pc15_seed0 gnn_v3_rpe1_pc15_seed1 gnn_v3_rpe1_pc15_seed2 gnn_v3_rpe1_pc15_seed3 gnn_v3_rpe1_pc15_seed4 &
eval_dataset 2 gnn_v3_k562_pc5_seed0 gnn_v3_k562_pc5_seed1 gnn_v3_k562_pc5_seed2 gnn_v3_k562_pc5_seed3 gnn_v3_k562_pc5_seed4 &
eval_dataset 3 gnn_v3_adamson_pc0p5_seed0 gnn_v3_adamson_pc0p5_seed1 gnn_v3_adamson_pc0p5_seed2 gnn_v3_adamson_pc0p5_seed3 gnn_v3_adamson_pc0p5_seed4 &
wait

echo "=== ALL DONE $(date +%T) ==="
