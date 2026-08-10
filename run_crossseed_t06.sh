#!/bin/bash
# run_crossseed_t06.sh
# Cross-seed G+ convergence subnetworks at thr_prob=0.6 for GNN v4 K562 (pc7).
# Phase 1: extraction in PARALLEL across GPUs 1/2/3 (GPU-bound, no API).
# Phase 2: enrichment SERIALLY (Speedrichr API is rate-limited; one at a time).
# seed4 already done (subnet2_t06_*) -> we (re)do all 5 for a uniform set.

cd ~/data/GPO-VAE
mkdir -p logs

SEEDS="0 1 2 3 4"
GPUS=(1 2 3)
PERT=622
THR=0.6

echo "=== PHASE 1: extraction (parallel) ==="
i=0
for s in $SEEDS; do
  g=${GPUS[$((i % 3))]}
  d=results/gnn_v4sweep_k562_pc7_seed${s}
  CUDA_VISIBLE_DEVICES=$g WANDB_MODE=disabled python subnetworks_v2.py \
    --experiment_path "$d" --pert_gene_len $PERT --devices 0 \
    --thr_prob $THR --wd_min 0.3 --min_parents 3 --max_parents 30 \
    --out_prefix "$d/subnet2_t06" > logs/cs_extract_seed${s}.log 2>&1 &
  i=$((i+1))
  while [ "$(jobs -r | wc -l)" -ge 3 ]; do sleep 5; done
done
wait
echo "extraction done"

echo "=== PHASE 2: enrichment (serial) ==="
for s in $SEEDS; do
  d=results/gnn_v4sweep_k562_pc7_seed${s}
  echo "  enriching seed ${s}..."
  python enrich_subnets_bg.py \
    --conv "$d/subnet2_t06_convergence.csv" \
    --grn "$d/grn.csv" --pert_gene_len $PERT \
    --out "$d/subnet2_t06_enrichment_bg.csv" --fdr 0.05 \
    > logs/cs_enrich_seed${s}.log 2>&1
done
echo "ALL DONE" > logs/crossseed_t06_done.flag
echo "complete"