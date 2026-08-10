"""
extract_edge_wd.py

Extract per-edge Wasserstein distances from a trained GPO-VAE / GNN-v4 checkpoint,
for the Section-3.5-style subnetwork / pathway analysis.

The repo's predictor._evaluate_network computes a Wasserstein distance per edge
(observational vs interventional expression of the child gene) but discards the
(parent, child) identity -- it only keeps np.mean(...). This script reuses the
exact same machinery but KEEPS the edge identity, so every G0 edge gets a
(parent, child, prob, WD, p_value) row that can be ranked and turned into
subnetworks.

It primes the predictor by calling statistical_evaluation once (which fills
self.expression_matrix / self.gene_to_index / self.gene_to_interventions via
preprocess_dataset(criteria='FC3') on the test split -- i.e. the corrected
pipeline), then walks the G0 edge list from grn.csv and recomputes WD per edge,
reusing predictor.get_observational / get_interventional verbatim.

USAGE (run from repo root, same env as eval.py):

    CUDA_VISIBLE_DEVICES=1 WANDB_MODE=disabled python extract_edge_wd.py \
        --experiment_path results/gpo_vae_replogle_seed0 \
        --pert_gene_len 622 \
        --devices 0 \
        --out results/gpo_vae_replogle_seed0/edge_wd_g0.csv

Notes:
  * --pert_gene_len is the G0 boundary: K562=622, RPE1=383 (perturbed genes come
    first in grn.csv row/col order). Adamson is tiny; not recommended.
  * grn.csv is read from <experiment_path>/grn.csv (written by eval.py).
  * Only edges whose BOTH endpoints survive in the primed gene_to_interventions
    map get a WD; others are skipped and counted (FC3 filtering can drop some).
  * Output is a CSV sorted by WD descending.
"""
import argparse
import os
import sys
from os.path import basename, join, splitext

import numpy as np
import pandas as pd
import torch
import scipy.stats

# mirror eval.py's path setup so imports resolve from repo root
os.chdir('/'.join(os.path.abspath(__file__).split('/')[:-1]))
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-1]))

from gpo_vae.models.utils.perturbation_lightning_module import (
    TrainConfigPerturbationLightningModule,
)


def load_checkpoint(checkpoint_path, devices):
    if devices is None:
        return TrainConfigPerturbationLightningModule.load_from_checkpoint(checkpoint_path)
    return TrainConfigPerturbationLightningModule.load_from_checkpoint(
        checkpoint_path,
        map_location=lambda storage, loc: storage.cuda(devices)
        if torch.cuda.is_available() else storage,
    )


def find_best_checkpoint(experiment_path):
    ckpt_dir = join(experiment_path, "checkpoints")
    names = os.listdir(ckpt_dir)
    best = [x for x in names if x[:4] == "best"]
    assert len(best) == 1, f"expected exactly one best*.ckpt, found {best}"
    return join(ckpt_dir, best[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment_path", required=True)
    ap.add_argument("--pert_gene_len", type=int, required=True,
                    help="G0 boundary: K562=622, RPE1=383")
    ap.add_argument("--devices", type=int, default=0)
    ap.add_argument("--thr_prob", type=float, default=0.5,
                    help="edge probability threshold (default 0.5, paper)")
    ap.add_argument("--p_value_threshold", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = args.out or join(args.experiment_path, "edge_wd_g0.csv")

    # ---- load model + data, prime predictor ----
    ckpt = find_best_checkpoint(args.experiment_path)
    print(f"[load] {ckpt}")
    lm = load_checkpoint(ckpt, args.devices)
    data_module = lm.get_data_module()
    predictor = lm.predictor

    # Prime predictor.expression_matrix / gene maps via one statistical_evaluation
    # call. We pass an EMPTY network so the internal _evaluate_network loops do
    # almost nothing -- we only want the side effect of populating
    # self.expression_matrix, self.gene_to_index, self.gene_to_interventions,
    # self.index_to_gene (predictor.py lines ~463-476).
    print("[prime] running statistical_evaluation(network=[]) to populate maps")
    predictor.statistical_evaluation(
        data_module=data_module,
        network=[],
        n_particles=100,
        check_false_omission_rate=False,
        omission_estimation_size=0,
        p_value_threshold=args.p_value_threshold,
    )
    print(f"[prime] gene universe (FC3): {len(predictor.gene_names)} genes; "
          f"{len(predictor.gene_to_interventions)} interventions")

    # ---- read grn.csv, build G0 edge list as gene-name pairs ----
    grn_path = join(args.experiment_path, "grn.csv")
    grn = pd.read_csv(grn_path, index_col=0)
    genes = grn.index.tolist()
    P = grn.values.copy()
    np.fill_diagonal(P, 0.0)
    PERT = args.pert_gene_len
    assert PERT <= len(genes), f"pert_gene_len {PERT} > matrix size {len(genes)}"

    sub = P[:PERT, :PERT]                      # G0 x G0
    ii, jj = np.nonzero(sub > args.thr_prob)   # directed edges, no self-loop
    print(f"[edges] G0 edges > {args.thr_prob}: {len(ii)}")

    # ---- WD-keeping loop (reuses predictor.get_observational/get_interventional) ----
    g2i = predictor.gene_to_interventions   # gene name -> cell indices (parent must be here)
    g2idx = predictor.gene_to_index         # gene name -> column (child must be here)

    rows = []
    skipped = 0
    n = len(ii)
    for k in range(n):
        pi, ci = int(ii[k]), int(jj[k])
        parent, child = genes[pi], genes[ci]
        prob = float(sub[pi, ci])

        # both endpoints must exist in the primed maps (FC3 may drop some)
        if parent not in g2i or child not in g2idx:
            skipped += 1
            continue
        try:
            obs = predictor.get_observational(child)
            interv = predictor.get_interventional(child, parent)
        except KeyError:
            skipped += 1
            continue
        if len(obs) == 0 or len(interv) == 0:
            skipped += 1
            continue

        wd = scipy.stats.wasserstein_distance(obs, interv)
        try:
            pval = scipy.stats.mannwhitneyu(obs, interv)[1]
        except ValueError:
            pval = np.nan
        rows.append({"parent": parent, "child": child,
                     "prob": prob, "wd": wd, "p_value": pval,
                     "parent_idx": pi, "child_idx": ci})

        if (k + 1) % 200 == 0:
            print(f"  ...{k+1}/{n} edges processed")

    df = pd.DataFrame(rows).sort_values("wd", ascending=False).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"\n[done] {len(df)} edges with WD ({skipped} skipped, not in FC3 universe)")
    print(f"[done] saved -> {out_path}")
    if len(df):
        print("\nTop 15 G0 edges by WD:")
        print(df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()