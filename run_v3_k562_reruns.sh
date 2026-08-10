#!/bin/bash
GUIDE=gpo_vae/models/gpo_vae/guides/gnn_guide.py
V3=gpo_vae/models/gpo_vae/guides/gnn_guide.py.v3
for s in 0 2 3; do
  cp "$V3" "$GUIDE"   # re-assert v3 immediately before launch
  if ! grep -q "v3 no-detach ACTIVE" "$GUIDE"; then echo "ABORT seed$s: not v3"; exit 1; fi
  name=gnn_v3_k562_pc5_seed${s}
  echo "[GPU2] RERUN START $name $(date +%T)"
  CUDA_VISIBLE_DEVICES=2 python train_replogle.py --config ./demo/${name}.yaml \
    > logs/train_${name}_rerun.log 2>&1
  echo "[GPU2] RERUN DONE  $name $(date +%T)"
done
echo "ALL K562 RERUNS DONE $(date +%T)"
