#!/bin/bash
run_queue () {
  local gpu=$1; shift
  for d in "$@"; do
    name=$(basename "$d")
    echo "[GPU $gpu] START $name $(date +%T)"
    CUDA_VISIBLE_DEVICES=$gpu WANDB_MODE=disabled python eval.py \
      --experiment_path "$d" \
      --perturbseq --batch_size 128 --ate_n_particles 2500 \
      --devices 0 --thr 3 \
      > "logs/alleval_${name}.log" 2>&1
    echo "[GPU $gpu] DONE  $name $(date +%T)"
  done
}

# GPU 1: all RPE1 (light)
run_queue 1 \
  results/gpo_vae_rpe1_seed0-3 results/gpo_vae_rpe1_seed1-3 results/gpo_vae_rpe1_seed2-3 \
  results/gpo_vae_rpe1_seed3-3 results/gpo_vae_rpe1_seed4-3 \
  results/gnn_vae_rpe1_pc15_seed0-2 results/gnn_v4_rpe1_pc15_seed1 results/gnn_v4_rpe1_pc15_seed2 \
  results/gnn_v4_rpe1_pc15_seed3 results/gnn_v4_rpe1_pc15_seed4 &

# GPU 2: all K562 (heavy ~43GB, serial, GPU2 is the only clear card)
run_queue 2 \
  results/gpo_vae_replogle_seed0 results/gpo_vae_replogle_seed1 results/gpo_vae_replogle_seed2 \
  results/gpo_vae_replogle_seed3 results/gpo_vae_replogle_seed4 \
  results/gnn_vae_k562_pc5_seed0-3 results/gnn_v4_k562_pc5_seed1 results/gnn_v4_k562_pc5_seed2 \
  results/gnn_v4_k562_pc5_seed3 results/gnn_v4_k562_pc5_seed4 &

# GPU 3: all Adamson (light)
run_queue 3 \
  results/gpo_vae_adamson_pc1_seed0 results/gpo_vae_adamson_pc1_seed1 results/gpo_vae_adamson_pc1_seed2 \
  results/gpo_vae_adamson_pc1_seed3 results/gpo_vae_adamson_pc1_seed4 \
  results/gnn_v4_adamson_pc0p5_seed0 results/gnn_v4_adamson_pc0p5_seed1 results/gnn_v4_adamson_pc0p5_seed2 \
  results/gnn_v4_adamson_pc0p5_seed3 results/gnn_v4_adamson_pc0p5_seed4 &

wait
echo "ALL EVALS COMPLETE $(date +%T)"
