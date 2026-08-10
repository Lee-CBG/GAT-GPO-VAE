"""
subnetworks_v2.py

Section-3.5-style subnetwork extraction from a trained GPO-VAE / GNN-v4 checkpoint,
using per-edge Wasserstein distance (WD) as causal-strength ranking.

Two subnetwork types:

  (a) G0->G0 convergence hubs (within the perturbed block)
      A child gene with >= hub_min_parents high-WD perturbed parents.
      (Connected-component reporting is also emitted, but on K562 the strong-WD
       G0 edges fuse into one functional supercore -- the hubs are the useful unit.)

  (b) Convergence subnetworks via a shared EXTENDED gene  <-- the faithful,
      data-supported version of the paper's "perturbed genes linked through an
      extended gene" idea. Directed p->e->q bridges do NOT exist as confident
      paths (extended genes are sinks, not pass-throughs), so instead we group
      perturbed genes that CONVERGE on a common extended gene:
          subnetwork(e) = { p in G0 : prob(p->e) > thr_prob and WD(p->e) >= wd_min }
      Filters:
        - drop mega-sink extended genes (n_parents > max_parents, e.g. RPL29=544)
        - require min_parents <= n_parents <= max_parents
        - rank parents by WD, keep subnetwork if >= min_parents survive wd_min
      Each subnetwork's gene set { e } U parents goes to DAVID/Enrichr.

Loads model once, primes predictor (FC3 + test split), computes WD via the repo's
own predictor.get_observational / get_interventional.

USAGE:
    CUDA_VISIBLE_DEVICES=1 WANDB_MODE=disabled python subnetworks_v2.py \
        --experiment_path results/gpo_vae_replogle_seed0 \
        --pert_gene_len 622 --devices 0 \
        --wd_min 0.3 --min_parents 3 --max_parents 30 \
        --out_prefix results/gpo_vae_replogle_seed0/subnet2

Outputs:
    <out_prefix>_g0_edges.csv        all G0->G0 edges with WD (ranked)
    <out_prefix>_hubs.txt            (a) G0 convergence hubs
    <out_prefix>_convergence.csv     (b) one row per (extended_gene, parent) kept
    <out_prefix>_convergence.txt     (b) readable, one block per extended gene
    <out_prefix>_genesets.txt        gene lists per subnetwork for DAVID/Enrichr
"""
import argparse
import os
import sys
from os.path import join

import numpy as np
import pandas as pd
import scipy.stats
import torch
import networkx as nx

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
    # (a)
    ap.add_argument("--hub_min_parents", type=int, default=4)
    ap.add_argument("--topk_g0", type=int, default=300)
    # (b)
    ap.add_argument("--wd_min", type=float, default=0.3)
    ap.add_argument("--min_parents", type=int, default=3)
    ap.add_argument("--max_parents", type=int, default=30,
                    help="drop mega-sink extended genes with more parents than this")
    ap.add_argument("--out_prefix", default=None)
    args = ap.parse_args()

    pre = args.out_prefix or join(args.experiment_path, "subnet2")
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
    n = len(genes)

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

    # ---------------------------------------------------------------
    # (a) G0 convergence hubs
    # ---------------------------------------------------------------
    sub = P[:PERT, :PERT]
    ii, jj = np.nonzero(sub > args.thr_prob)
    g0_edges = []
    for i, j in zip(ii, jj):
        w = wd(genes[i], genes[j])
        if w is not None:
            g0_edges.append((genes[i], genes[j], float(sub[i, j]), w))
    g0_df = pd.DataFrame(g0_edges, columns=["parent", "child", "prob", "wd"]) \
              .sort_values("wd", ascending=False).reset_index(drop=True)
    g0_df.to_csv(f"{pre}_g0_edges.csv", index=False)

    top = g0_df.head(args.topk_g0)
    G = nx.DiGraph()
    for _, r in top.iterrows():
        G.add_edge(r["parent"], r["child"], wd=r["wd"], prob=r["prob"])
    hubs = [(c, list(G.predecessors(c))) for c in G.nodes()
            if G.in_degree(c) >= args.hub_min_parents]
    hubs.sort(key=lambda x: len(x[1]), reverse=True)
    with open(f"{pre}_hubs.txt", "w") as f:
        f.write(f"# (a) G0->G0 convergence hubs (top-{args.topk_g0} WD edges, "
                f">= {args.hub_min_parents} parents)\n# {ckpt}\n\n")
        for child, parents in hubs:
            f.write(f"### {child}  <- {len(parents)} parents\n")
            for p in sorted(parents, key=lambda p: -G[p][child]['wd']):
                f.write(f"   {p:12s} -> {child:12s}  wd={G[p][child]['wd']:.3f}\n")
            f.write("\n")
    print(f"[a] {len(hubs)} G0 hubs -> {pre}_hubs.txt")

    # ---------------------------------------------------------------
    # (b) convergence subnetworks via shared extended gene
    # ---------------------------------------------------------------
    g0_gp = P[:PERT, PERT:]                       # G0 -> G+ probs
    conv_bool = g0_gp > args.thr_prob
    n_parents_each = conv_bool.sum(0)             # per extended gene

    conv_rows = []
    kept_subnets = {}                             # extended_gene -> list of (parent, prob, wd)
    n_ext = n - PERT
    for b in range(n_ext):
        np_count = int(n_parents_each[b])
        if np_count < args.min_parents or np_count > args.max_parents:
            continue
        ext_gene = genes[PERT + b]
        parents_idx = np.nonzero(conv_bool[:, b])[0]
        scored = []
        for a in parents_idx:
            p_gene = genes[a]
            w = wd(p_gene, ext_gene)
            if w is not None and w >= args.wd_min:
                scored.append((p_gene, float(g0_gp[a, b]), w))
        if len(scored) >= args.min_parents:
            scored.sort(key=lambda x: -x[2])
            kept_subnets[ext_gene] = scored
            for p_gene, prob, w in scored:
                conv_rows.append({"extended": ext_gene, "parent": p_gene,
                                  "prob": prob, "wd": w, "n_parents": len(scored)})

    conv_df = pd.DataFrame(conv_rows)
    if len(conv_df):
        # rank subnetworks by median WD of their parents
        med = conv_df.groupby("extended")["wd"].median().rename("median_wd")
        conv_df = conv_df.merge(med, on="extended")
        conv_df = conv_df.sort_values(["median_wd", "extended", "wd"],
                                      ascending=[False, True, False]).reset_index(drop=True)
    conv_df.to_csv(f"{pre}_convergence.csv", index=False)

    with open(f"{pre}_convergence.txt", "w") as f:
        f.write(f"# (b) convergence subnetworks via shared extended gene\n")
        f.write(f"# wd_min={args.wd_min}, parents in [{args.min_parents},{args.max_parents}]\n# {ckpt}\n\n")
        # order by median WD
        order = sorted(kept_subnets.items(),
                       key=lambda kv: -np.median([x[2] for x in kv[1]]))
        for ext_gene, scored in order:
            mwd = np.median([x[2] for x in scored])
            f.write(f"### {ext_gene}  <- {len(scored)} perturbed parents  (median WD {mwd:.3f})\n")
            for p_gene, prob, w in scored:
                f.write(f"   {p_gene:12s} -> {ext_gene:12s}  wd={w:.3f}  prob={prob:.3f}\n")
            f.write("\n")
    print(f"[b] {len(kept_subnets)} convergence subnetworks -> {pre}_convergence.txt")

    # ---------------------------------------------------------------
    # gene sets for DAVID / Enrichr
    # ---------------------------------------------------------------
    with open(f"{pre}_genesets.txt", "w") as f:
        f.write("# Gene sets for DAVID / Enrichr\n\n")
        for child, parents in hubs:
            gs = sorted(set(parents) | {child})
            f.write(f"## (a) hub {child} ({len(gs)} genes)\n")
            f.write("\n".join(gs) + "\n\n")
        order = sorted(kept_subnets.items(),
                       key=lambda kv: -np.median([x[2] for x in kv[1]]))
        for ext_gene, scored in order:
            gs = sorted({ext_gene} | {p for p, _, _ in scored})
            f.write(f"## (b) convergence via {ext_gene} ({len(gs)} genes)\n")
            f.write("\n".join(gs) + "\n\n")
    print(f"[sets] -> {pre}_genesets.txt")

    # console preview
    print("\n=== (a) top G0 hubs ===")
    for child, parents in hubs[:6]:
        print(f"  {child} <- ({len(parents)}) {sorted(parents)[:8]}{'...' if len(parents)>8 else ''}")
    print("\n=== (b) top convergence subnetworks (by median WD) ===")
    order = sorted(kept_subnets.items(),
                   key=lambda kv: -np.median([x[2] for x in kv[1]]))
    for ext_gene, scored in order[:10]:
        mwd = np.median([x[2] for x in scored])
        plist = [p for p, _, _ in scored]
        print(f"  {ext_gene:12s} (n={len(scored)}, medWD={mwd:.2f}): {plist[:8]}{'...' if len(plist)>8 else ''}")


if __name__ == "__main__":
    main()