#!/bin/bash
GUIDE=gpo_vae/models/gpo_vae/guides/gnn_guide.py

if ! grep -q "^        edge_weight = torch.sigmoid(q_mask_logits)  # v3 no-detach ACTIVE" "$GUIDE"; then
  echo "ABORT: live guide is not v3."; exit 1
fi
echo "Confirmed live guide = v3. Start $(date +%T)"

launch () {  # gpu script cfg
  local name=$(basename "$3" .yaml)
  echo "[GPU $1] START $name $(date +%T)"
  CUDA_VISIBLE_DEVICES=$1 python $2 --config ./demo/$3 > "logs/train_${name}.log" 2>&1
  echo "[GPU $1] DONE  $name $(date +%T)"
}

# GPU 1: all 5 RPE1
for s in 0 1 2 3 4; do launch 1 train_rpe1.py     gnn_v3_rpe1_pc15_seed${s}.yaml & done
# GPU 2: all 5 K562
for s in 0 1 2 3 4; do launch 2 train_replogle.py gnn_v3_k562_pc5_seed${s}.yaml & done
# GPU 3: all 5 Adamson
for s in 0 1 2 3 4; do launch 3 train_adamson.py  gnn_v3_adamson_pc0p5_seed${s}.yaml & done

wait
echo "ALL V3 TRAINING COMPLETE $(date +%T)"
cp gpo_vae/models/gpo_vae/guides/gnn_guide.py.v4 "$GUIDE"
echo "RESTORED guide -> v4 (detach ON) $(date +%T)"
grep -n "edge_weight = torch.sigmoid" "$GUIDE" | tail -1
