"""
subnetworks.py

Build Section-3.5-style subnetworks from a trained GPO-VAE / GNN-v4 checkpoint,
using per-edge Wasserstein distance (WD) as the causal-strength ranking.

Produces TWO kinds of subnetworks:

  (a) G0->G0 subnetworks
      - keep the top-K highest-WD G0->G0 edges
      - report weakly-connected components (>= min_component_size nodes)
      - report convergence hubs (a child with >= hub_min_parents high-WD parents,
        e.g. the proteasome->WDR36 cluster)

  (b) Multi-hop G0->G+->G0 motifs (the paper's stated criterion i), GUARDED so it
      does not drown in the saturated G+->G0 block:
        - the G0->G+ leg (sparse, selective) must have prob > thr_prob AND
          WD >= wd_min
        - the G+->G0 leg must ALSO have WD >= wd_min
        - the extended intermediate must be selective: out-degree (>thr_prob)
          <= max_intermediate_outdeg  (drops broadcast hubs like ATP1A1)
        - no direct G0->G0 edge between the two perturbed endpoints (paper crit ii)
      ranked by min(WD_leg1, WD_leg2) -- the weaker leg bounds the path strength.

This script loads the model ONCE, primes the predictor (FC3 + test split, the
corrected pipeline), and computes WD for every edge it needs via the repo's own
predictor.get_observational / get_interventional.

USAGE (repo root, eval.py env):

    CUDA_VISIBLE_DEVICES=1 WANDB_MODE=disabled python subnetworks.py \
        --experiment_path results/gpo_vae_replogle_seed0 \
        --pert_gene_len 622 --devices 0 \
        --topk_g0 150 --wd_min 0.3 \
        --out_prefix results/gpo_vae_replogle_seed0/subnet

Outputs:
    <out_prefix>_components.txt     human-readable G0 components + hubs
    <out_prefix>_g0_edges.csv       the top-K G0->G0 edges used (parent,child,prob,wd)
    <out_prefix>_multihop.csv       guarded G0->G+->G0 motifs, ranked
    <out_prefix>_genesets.txt       gene lists per subnetwork, ready for DAVID/Enrichr
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
    # (a) params
    ap.add_argument("--topk_g0", type=int, default=150,
                    help="number of top-WD G0->G0 edges to build components from")
    ap.add_argument("--min_component_size", type=int, default=3)
    ap.add_argument("--hub_min_parents", type=int, default=3)
    # (b) params
    ap.add_argument("--wd_min", type=float, default=0.3,
                    help="min WD on BOTH legs of a multi-hop path")
    ap.add_argument("--max_intermediate_outdeg", type=int, default=50,
                    help="drop extended intermediates broadcasting to more than this many G0 genes")
    ap.add_argument("--multihop_max_paths", type=int, default=300,
                    help="cap number of candidate paths scored (by G0->G+ prob) to bound runtime")
    ap.add_argument("--out_prefix", default=None)
    args = ap.parse_args()

    pre = args.out_prefix or join(args.experiment_path, "subnet")
    PERT = args.pert_gene_len

    # ---- load + prime ----
    ckpt = find_best_checkpoint(args.experiment_path)
    print(f"[load] {ckpt}")
    lm = load_checkpoint(ckpt, args.devices)
    data_module = lm.get_data_module()
    predictor = lm.predictor
    print("[prime] statistical_evaluation(network=[]) to populate maps")
    predictor.statistical_evaluation(
        data_module=data_module, network=[], n_particles=100,
        check_false_omission_rate=False, omission_estimation_size=0,
        p_value_threshold=args.p_value_threshold,
    )
    g2i = predictor.gene_to_interventions
    g2idx = predictor.gene_to_index
    print(f"[prime] FC3 universe: {len(predictor.gene_names)} genes, "
          f"{len(g2i)} interventions")

    # ---- topology from grn.csv ----
    grn = pd.read_csv(join(args.experiment_path, "grn.csv"), index_col=0)
    genes = grn.index.tolist()
    P = grn.values.copy()
    np.fill_diagonal(P, 0.0)
    n = len(genes)
    is_pert = np.zeros(n, bool); is_pert[:PERT] = True

    def wd(parent, child):
        """per-edge WD via repo machinery; None if endpoints not in FC3 universe."""
        if parent not in g2i or child not in g2idx:
            return None
        try:
            obs = predictor.get_observational(child)
            iv = predictor.get_interventional(child, parent)
        except KeyError:
            return None
        if len(obs) == 0 or len(iv) == 0:
            return None
        return float(scipy.stats.wasserstein_distance(obs, iv))

    # =====================================================================
    # (a) G0 -> G0 subnetworks
    # =====================================================================
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

    # weakly-connected components
    comps = [c for c in nx.weakly_connected_components(G)
             if len(c) >= args.min_component_size]
    comps.sort(key=len, reverse=True)

    # convergence hubs (child with many high-WD parents)
    hubs = []
    for child in G.nodes():
        parents = list(G.predecessors(child))
        if len(parents) >= args.hub_min_parents:
            hubs.append((child, parents))
    hubs.sort(key=lambda x: len(x[1]), reverse=True)

    with open(f"{pre}_components.txt", "w") as f:
        f.write(f"# G0->G0 subnetworks from top-{args.topk_g0} WD edges\n")
        f.write(f"# checkpoint: {ckpt}\n\n")
        f.write(f"## Weakly-connected components (>= {args.min_component_size} nodes): "
                f"{len(comps)}\n\n")
        for ci, c in enumerate(comps):
            sg = G.subgraph(c)
            f.write(f"### Component {ci+1}  ({len(c)} genes, {sg.number_of_edges()} edges)\n")
            f.write(f"genes: {sorted(c)}\n")
            for u, v, d in sorted(sg.edges(data=True), key=lambda e: -e[2]['wd']):
                f.write(f"   {u:12s} -> {v:12s}  wd={d['wd']:.3f}  prob={d['prob']:.3f}\n")
            f.write("\n")
        f.write(f"## Convergence hubs (>= {args.hub_min_parents} high-WD parents):\n\n")
        for child, parents in hubs:
            f.write(f"### {child}  <- {len(parents)} parents\n")
            for p in sorted(parents, key=lambda p: -G[p][child]['wd']):
                f.write(f"   {p:12s} -> {child:12s}  wd={G[p][child]['wd']:.3f}\n")
            f.write("\n")
    print(f"[a] {len(comps)} components, {len(hubs)} hubs -> {pre}_components.txt")

    # =====================================================================
    # (b) guarded multi-hop  G0 -> G+ -> G0
    # =====================================================================
    # selective G0->G+ edges (sparse leg), ranked by prob, capped for runtime
    g0_gplus = P[:PERT, PERT:]
    pi, pj = np.nonzero(g0_gplus > args.thr_prob)
    order = np.argsort(g0_gplus[pi, pj])[::-1][:args.multihop_max_paths]
    pi, pj = pi[order], pj[order]

    # precompute extended-gene out-degree to G0 (selectivity guard)
    gplus_g0 = P[PERT:, :PERT]
    ext_outdeg = (gplus_g0 > args.thr_prob).sum(axis=1)  # per extended gene

    motifs = []
    for a, b in zip(pi, pj):
        ext_local = b                      # column index within G+ block
        ext_global = PERT + ext_local
        if ext_outdeg[ext_local] > args.max_intermediate_outdeg:
            continue                       # broadcast hub -> drop
        p_gene = genes[a]                  # perturbed source
        e_gene = genes[ext_global]         # extended intermediate
        # leg 1 WD: p_gene -> e_gene
        wd1 = wd(p_gene, e_gene)
        if wd1 is None or wd1 < args.wd_min:
            continue
        # find perturbed children q of this extended gene with WD support
        children = np.nonzero(gplus_g0[ext_local] > args.thr_prob)[0]
        for q in children:
            q_gene = genes[q]
            if q_gene == p_gene:
                continue
            if P[a, q] > args.thr_prob:     # direct G0->G0 edge exists -> skip (crit ii)
                continue
            wd2 = wd(e_gene, q_gene)
            if wd2 is None or wd2 < args.wd_min:
                continue
            motifs.append({
                "perturbed_src": p_gene, "extended": e_gene, "perturbed_dst": q_gene,
                "wd_leg1": wd1, "wd_leg2": wd2, "path_strength": min(wd1, wd2),
                "ext_outdeg": int(ext_outdeg[ext_local]),
            })

    mh = pd.DataFrame(motifs)
    if len(mh):
        mh = mh.sort_values("path_strength", ascending=False).reset_index(drop=True)
    mh.to_csv(f"{pre}_multihop.csv", index=False)
    print(f"[b] {len(mh)} guarded multi-hop motifs -> {pre}_multihop.csv")

    # =====================================================================
    # gene sets for DAVID / Enrichr
    # =====================================================================
    with open(f"{pre}_genesets.txt", "w") as f:
        f.write("# Gene sets for DAVID / Enrichr (one set per line block)\n\n")
        for ci, c in enumerate(comps):
            f.write(f"## G0 component {ci+1} ({len(c)} genes)\n")
            f.write("\n".join(sorted(c)) + "\n\n")
        if len(mh):
            # union of genes per extended intermediate (a multi-hop subnetwork)
            for ext, grp in mh.groupby("extended"):
                gs = sorted(set(grp["perturbed_src"]) | {ext} | set(grp["perturbed_dst"]))
                f.write(f"## multi-hop via {ext} ({len(gs)} genes)\n")
                f.write("\n".join(gs) + "\n\n")
    print(f"[sets] gene sets -> {pre}_genesets.txt")

    # console preview
    print("\n=== (a) top components ===")
    for ci, c in enumerate(comps[:5]):
        print(f"  component {ci+1}: {sorted(c)}")
    print("\n=== (a) top hubs ===")
    for child, parents in hubs[:5]:
        print(f"  {child} <- {sorted(parents)}")
    if len(mh):
        print("\n=== (b) top multi-hop motifs ===")
        print(mh.head(15).to_string(index=False))


if __name__ == "__main__":
    main()