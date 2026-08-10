"""
aggregate_subnets.py

Cross-seed aggregation of convergence subnetworks (subnetworks_v2.py output) for
GPO-VAE vs GNN-v4 on K562.

Answers two questions:
  (1) Module reproducibility: do the same biological modules recur across seeds?
      Matched by GENE-SET CONTENT, not by extended-gene label (the anchor gene is
      not stable across seeds; the biology is).
  (2) WD comparison: per module, mean +/- std of median-WD across seeds, GPO-VAE
      vs GNN-v4.

Anchor modules are defined by member genes (the recurring biology we identified).
For each seed and model, we scan that seed's convergence subnetworks and pick the
one whose PARENT set best overlaps each anchor (Jaccard). We record whether a
match exists (overlap >= min_overlap genes) and its median WD.

USAGE (repo root):
    python aggregate_subnets.py \
        --gpo_glob 'results/gpo_vae_replogle_seed{s}/subnet2_convergence.csv' \
        --gnn_glob 'results/gnn_v4sweep_k562_pc7_seed{s}/subnet2_convergence.csv' \
        --seeds 0 1 2 3 4

No GPU needed -- reads CSVs only.
"""
import argparse
import numpy as np
import pandas as pd

# Anchor modules defined by member genes (the recurring biology at thr=0.6).
# Matching is by gene-set CONTENT, so the anchor label and enrichment term do not
# matter -- we ask whether each seed has a convergence subnetwork whose PARENT set
# overlaps the module's defining genes.
ANCHORS = {
    "RNA_exosome/mRNA_decay": ["EXOSC2", "EXOSC8", "ZFC3H1", "PABPN1", "DLD", "CSE1L"],
    "Mito_transcription/biogenesis": ["SSBP1", "POLRMT", "PNPT1", "TEFM", "HSPA9", "DNAJA3"],
    "Translation/eIF2_stress": ["EIF2S1", "PHB", "PHB2", "HSPA9", "DNAJC19", "TIMM23B"],
    "PolII_transcription/snRNA": ["INTS2", "INTS8", "CDC73", "PAF1", "CTR9", "SUPT6H"],
}


def best_match(conv_df, anchor_genes, min_overlap):
    """In one seed's convergence table, find the subnetwork whose parent set best
    overlaps anchor_genes. Return (extended, overlap_n, jaccard, median_wd) or None."""
    if conv_df is None or len(conv_df) == 0:
        return None
    anchor = set(anchor_genes)
    best = None
    for ext, grp in conv_df.groupby("extended"):
        parents = set(grp["parent"])
        ov = len(parents & anchor)
        if ov == 0:
            continue
        jac = ov / len(parents | anchor)
        medwd = float(grp["wd"].median())
        cand = (ext, ov, jac, medwd, len(parents))
        if best is None or ov > best[1] or (ov == best[1] and jac > best[2]):
            best = cand
    if best is None or best[1] < min_overlap:
        return None
    return best


def load(glob_pattern, seeds):
    out = {}
    for s in seeds:
        path = glob_pattern.format(s=s)
        try:
            out[s] = pd.read_csv(path)
        except Exception as e:
            print(f"  [warn] could not read {path}: {e}")
            out[s] = None
    return out


def summarize(name, tables, seeds, min_overlap):
    print(f"\n================  {name}  ================")
    rows = []
    for mod, genes in ANCHORS.items():
        recovered, wds, overlaps, exts = 0, [], [], []
        for s in seeds:
            m = best_match(tables.get(s), genes, min_overlap)
            if m is not None:
                recovered += 1
                exts.append(m[0]); overlaps.append(m[1]); wds.append(m[3])
        if wds:
            print(f"  {mod:22s} recovered {recovered}/{len(seeds)} seeds | "
                  f"median-WD {np.mean(wds):.3f} ± {np.std(wds):.3f} | "
                  f"overlap {np.mean(overlaps):.1f} genes | anchors: {sorted(set(exts))}")
        else:
            print(f"  {mod:22s} recovered 0/{len(seeds)} seeds")
        rows.append({"module": mod, "model": name,
                     "recovered": recovered, "n_seeds": len(seeds),
                     "wd_mean": np.mean(wds) if wds else np.nan,
                     "wd_std": np.std(wds) if wds else np.nan,
                     "overlap_mean": np.mean(overlaps) if overlaps else np.nan})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpo_glob", required=True)
    ap.add_argument("--gnn_glob", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--min_overlap", type=int, default=2,
                    help="min anchor genes overlapping to count as recovered")
    ap.add_argument("--out", default="results/subnet_module_summary.csv")
    args = ap.parse_args()

    gpo = load(args.gpo_glob, args.seeds)
    gnn = load(args.gnn_glob, args.seeds)

    g_df = summarize("GPO-VAE", gpo, args.seeds, args.min_overlap)
    n_df = summarize("GNN-v4", gnn, args.seeds, args.min_overlap)

    both = pd.concat([g_df, n_df], ignore_index=True)
    both.to_csv(args.out, index=False)

    # head-to-head WD table
    print("\n================  WD comparison (median-WD across seeds)  ================")
    print(f"{'module':22s} {'GPO-VAE':>18s} {'GNN-v4':>18s} {'Δ(v4-gpo)':>10s}")
    for mod in ANCHORS:
        g = g_df[g_df.module == mod].iloc[0]
        nn = n_df[n_df.module == mod].iloc[0]
        gtxt = f"{g.wd_mean:.3f}±{g.wd_std:.3f}" if not np.isnan(g.wd_mean) else "  --  "
        ntxt = f"{nn.wd_mean:.3f}±{nn.wd_std:.3f}" if not np.isnan(nn.wd_mean) else "  --  "
        if not np.isnan(g.wd_mean) and not np.isnan(nn.wd_mean):
            d = f"{nn.wd_mean - g.wd_mean:+.3f}"
        else:
            d = "  --  "
        print(f"{mod:22s} {gtxt:>18s} {ntxt:>18s} {d:>10s}")
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()