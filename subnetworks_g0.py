"""
subnetworks_g0.py

G0->G0 convergence subnetworks: the clean-substrate counterpart to
subnetworks_v2.py. Here BOTH the anchor and its converging parents are perturbed
(G0) genes -- all protein-coding, all properly GRN-supervised, no G+ saturation,
no non-coding anchors. This is the defensible substrate for testing whether the
biological modules (mito / translation / exosome / etc.) are real properties of
the GRN or artifacts of using extended-gene (G+) anchors.

Definition (mirrors the G+ convergence version):
  subnetwork(a) = { p in G0 : p != a, prob(p->a) > thr_prob and WD(p->a) >= wd_min }
  kept if min_parents <= |parents| <= max_parents
Mega-sink guard: drop anchors with more than max_parents converging G0 parents
(prevents the ribosome/proteasome supercore from forming one giant blob).

Loads model once, primes predictor (FC3 + test split), computes WD via the repo's
predictor.get_observational / get_interventional. Same machinery as subnetworks_v2.

USAGE:
    CUDA_VISIBLE_DEVICES=1 WANDB_MODE=disabled python subnetworks_g0.py \
        --experiment_path results/gnn_v4sweep_k562_pc7_seed4 \
        --pert_gene_len 622 --devices 0 \
        --wd_min 0.3 --min_parents 3 --max_parents 30 \
        --out_prefix results/gnn_v4sweep_k562_pc7_seed4/subnet_g0

Outputs:
    <out_prefix>_convergence.csv   one row per (anchor, parent) kept
    <out_prefix>_convergence.txt   readable, one block per anchor
    <out_prefix>_genesets.txt      gene lists per subnetwork for enrichment
"""
import argparse
import os
import sys
from os.path import join

import numpy as np
import pandas as pd
import scipy.stats
import torch

os.chdir('/'.join(os.path.abspath(__file__).split('/')[:-1]))
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-1]))

from gpo_vae.models.utils.perturbation_lightning_module import (
    TrainConfigPerturbationLightningModule,
)


def load_checkpoint(path, devices):
    if devices is None:
        return TrainConfigPerturbationLightningModule.load_from_checkpoint(path)
    return TrainConfigPerturbationLightningModule.load_from_checkpoint(
        path,
        map_location=lambda storage, loc: storage.cuda(devices)
        if torch.cuda.is_available() else storage,
    )


def find_best_checkpoint(experiment_path):
    ckpt_dir = join(experiment_path, "checkpoints")
    best = [x for x in os.listdir(ckpt_dir) if x[:4] == "best"]
    assert len(best) == 1, f"expected one best*.ckpt, found {best}"
    return join(ckpt_dir, best[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment_path", required=True)
    ap.add_argument("--pert_gene_len", type=int, required=True)
    ap.add_argument("--devices", type=int, default=0)
    ap.add_argument("--thr_prob", type=float, default=0.5)
    ap.add_argument("--p_value_threshold", type=float, default=0.05)
    ap.add_argument("--wd_min", type=float, default=0.3)
    ap.add_argument("--min_parents", type=int, default=3)
    ap.add_argument("--max_parents", type=int, default=30)
    ap.add_argument("--out_prefix", default=None)
    args = ap.parse_args()

    pre = args.out_prefix or join(args.experiment_path, "subnet_g0")
    PERT = args.pert_gene_len

    ckpt = find_best_checkpoint(args.experiment_path)
    print(f"[load] {ckpt}")
    lm = load_checkpoint(ckpt, args.devices)
    data_module = lm.get_data_module()
    predictor = lm.predictor
    print("[prime] statistical_evaluation(network=[])")
    predictor.statistical_evaluation(
        data_module=data_module, network=[], n_particles=100,
        check_false_omission_rate=False, omission_estimation_size=0,
        p_value_threshold=args.p_value_threshold,
    )
    g2i = predictor.gene_to_interventions
    g2idx = predictor.gene_to_index
    print(f"[prime] FC3 universe: {len(predictor.gene_names)} genes, {len(g2i)} interventions")

    grn = pd.read_csv(join(args.experiment_path, "grn.csv"), index_col=0)
    genes = grn.index.tolist()
    P = grn.values.copy()
    np.fill_diagonal(P, 0.0)

    _wd_cache = {}
    def wd(parent, child):
        key = (parent, child)
        if key in _wd_cache:
            return _wd_cache[key]
        if parent not in g2i or child not in g2idx:
            _wd_cache[key] = None; return None
        try:
            o = predictor.get_observational(child)
            iv = predictor.get_interventional(child, parent)
        except KeyError:
            _wd_cache[key] = None; return None
        v = None if (len(o) == 0 or len(iv) == 0) else \
            float(scipy.stats.wasserstein_distance(o, iv))
        _wd_cache[key] = v
        return v

    # ---- G0 -> G0 convergence ----
    g0_block = P[:PERT, :PERT]                       # rows=parents, cols=anchors (both G0)
    conv_bool = g0_block > args.thr_prob
    n_parents_each = conv_bool.sum(0)                # per G0 anchor: # converging G0 parents

    conv_rows = []
    kept = {}                                        # anchor -> [(parent, prob, wd), ...]
    n_dropped_sink = 0
    for a in range(PERT):
        npar = int(n_parents_each[a])
        if npar < args.min_parents:
            continue
        if npar > args.max_parents:
            n_dropped_sink += 1
            continue
        anchor = genes[a]
        parents_idx = np.nonzero(conv_bool[:, a])[0]
        scored = []
        for p in parents_idx:
            p_gene = genes[p]
            w = wd(p_gene, anchor)
            if w is not None and w >= args.wd_min:
                scored.append((p_gene, float(g0_block[p, a]), w))
        if len(scored) >= args.min_parents:
            scored.sort(key=lambda x: -x[2])
            kept[anchor] = scored
            for p_gene, prob, w in scored:
                conv_rows.append({"anchor": anchor, "parent": p_gene,
                                  "prob": prob, "wd": w, "n_parents": len(scored)})

    conv_df = pd.DataFrame(conv_rows)
    if len(conv_df):
        med = conv_df.groupby("anchor")["wd"].median().rename("median_wd")
        conv_df = conv_df.merge(med, on="anchor").sort_values(
            ["median_wd", "anchor", "wd"], ascending=[False, True, False]).reset_index(drop=True)
    conv_df.to_csv(f"{pre}_convergence.csv", index=False)

    order = sorted(kept.items(), key=lambda kv: -np.median([x[2] for x in kv[1]]))
    with open(f"{pre}_convergence.txt", "w") as f:
        f.write(f"# G0->G0 convergence subnetworks (clean substrate, all protein-coding)\n")
        f.write(f"# wd_min={args.wd_min}, parents in [{args.min_parents},{args.max_parents}]\n")
        f.write(f"# {ckpt}\n")
        f.write(f"# dropped {n_dropped_sink} mega-sink anchors (> {args.max_parents} parents)\n\n")
        for anchor, scored in order:
            mwd = np.median([x[2] for x in scored])
            f.write(f"### {anchor}  <- {len(scored)} G0 parents  (median WD {mwd:.3f})\n")
            for p_gene, prob, w in scored:
                f.write(f"   {p_gene:12s} -> {anchor:12s}  wd={w:.3f}  prob={prob:.3f}\n")
            f.write("\n")

    with open(f"{pre}_genesets.txt", "w") as f:
        f.write("# G0->G0 convergence gene sets for enrichment\n\n")
        for anchor, scored in order:
            gs = sorted({anchor} | {p for p, _, _ in scored})
            f.write(f"## convergence via {anchor} ({len(gs)} genes)\n")
            f.write("\n".join(gs) + "\n\n")

    print(f"[g0] {len(kept)} G0->G0 convergence subnetworks "
          f"({n_dropped_sink} mega-sinks dropped) -> {pre}_convergence.txt")
    print("\n=== top G0->G0 convergence subnetworks (by median WD) ===")
    for anchor, scored in order[:15]:
        mwd = np.median([x[2] for x in scored])
        plist = [p for p, _, _ in scored]
        print(f"  {anchor:12s} (n={len(scored)}, medWD={mwd:.2f}): "
              f"{plist[:8]}{'...' if len(plist)>8 else ''}")


if __name__ == "__main__":
    main()