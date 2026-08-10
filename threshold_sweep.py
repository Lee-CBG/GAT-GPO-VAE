"""
threshold_sweep.py

Sensitivity sweep of the probability threshold on existing convergence subnetworks.
Reuses the WD already computed in subnet2_convergence.csv (raising the prob
threshold only REMOVES edges, so the higher-threshold set is a subset -- no model
reload, no recomputation). Shows how subnetwork count, sizes, and the headline
modules change as thr_prob goes 0.5 -> 0.6 -> 0.7 -> 0.8.

Doubles as the sensitivity analysis for the "arbitrary threshold" soft spot.

USAGE:
    python threshold_sweep.py \
        --conv results/gnn_v4sweep_k562_pc7_seed4/subnet2_convergence.csv \
        --thresholds 0.5 0.6 0.7 0.8 \
        --wd_min 0.3 --min_parents 3 --max_parents 30

Anchor column auto-detected (extended for G+, anchor for G0).
"""
import argparse
import numpy as np
import pandas as pd

# headline modules to track across thresholds, defined by member genes
ANCHOR_MODULES = {
    "exosome/mRNA-decay": ["EXOSC2", "EXOSC8", "ZFC3H1", "PABPN1", "WDR20", "RPS2"],
    "mito": ["SSBP1", "POLRMT", "PNPT1", "TEFM", "DNAJA3", "ZWINT", "ZUP1", "ZSWIM7"],
    "translation/eIF2": ["EIF2S1", "EIF2B5", "CHP1", "ACOT9", "TUBGCP5", "RFT1"],
    "p53/DNA-damage": ["ARL6IP4", "ZNF101", "POLR3C"],
}


def module_present(subnets, module_genes, min_overlap=2):
    """Is there any subnetwork whose member set overlaps the module by >= min_overlap?"""
    mg = set(module_genes)
    best = 0
    best_anchor = None
    for anchor, members in subnets.items():
        ov = len(set(members) & mg)
        if ov > best:
            best, best_anchor = ov, anchor
    return (best >= min_overlap), best, best_anchor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", required=True)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8])
    ap.add_argument("--wd_min", type=float, default=0.3)
    ap.add_argument("--min_parents", type=int, default=3)
    ap.add_argument("--max_parents", type=int, default=30)
    args = ap.parse_args()

    df = pd.read_csv(args.conv)
    acol = "extended" if "extended" in df.columns else "anchor"
    df = df[df["wd"] >= args.wd_min]   # WD filter held fixed

    print(f"anchor column: {acol} | WD>={args.wd_min} | parents in "
          f"[{args.min_parents},{args.max_parents}]\n")
    print(f"{'thr':>5s} {'#subnets':>9s} {'#edges':>8s} {'med_size':>9s} "
          f"{'  modules present (overlap>=2)'}")

    for thr in args.thresholds:
        sub = df[df["prob"] > thr]
        # rebuild subnetworks under this threshold
        subnets = {}
        for anchor, grp in sub.groupby(acol):
            members = list(grp["parent"])
            n = len(members)
            if args.min_parents <= n <= args.max_parents:
                subnets[anchor] = members + [anchor]
        sizes = [len(v) for v in subnets.values()]
        n_edges = int((sub.groupby(acol).size()
                       .pipe(lambda s: s[(s >= args.min_parents) & (s <= args.max_parents)])
                       .sum())) if len(subnets) else 0
        med_size = int(np.median(sizes)) if sizes else 0

        present = []
        for mod, genes in ANCHOR_MODULES.items():
            ok, ov, anc = module_present(subnets, genes)
            if ok:
                present.append(f"{mod}({anc}:{ov})")
        print(f"{thr:5.2f} {len(subnets):9d} {n_edges:8d} {med_size:9d}   "
              f"{', '.join(present) if present else '(none)'}")

    # detail at each threshold: which anchors survive
    print("\n--- surviving anchors per threshold ---")
    for thr in args.thresholds:
        sub = df[df["prob"] > thr]
        kept = []
        for anchor, grp in sub.groupby(acol):
            n = len(grp)
            if args.min_parents <= n <= args.max_parents:
                kept.append((anchor, n, float(grp["wd"].median())))
        kept.sort(key=lambda x: -x[2])
        print(f"\nthr={thr}: {len(kept)} subnetworks")
        for anchor, n, mwd in kept[:12]:
            print(f"   {anchor:12s} n={n:2d}  medWD={mwd:.2f}")


if __name__ == "__main__":
    main()