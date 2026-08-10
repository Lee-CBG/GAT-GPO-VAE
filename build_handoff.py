"""
build_handoff.py

Build a per-subnetwork handoff report for a biologist from:
  - subnet2_convergence.csv   (convergence subnetworks: extended gene + WD-ranked parents)
  - subnet2_enrichment_bg.csv (background-corrected KEGG/Reactome enrichment)

For each convergence subnetwork it emits a readable block with:
  - the extended (anchor) gene and its perturbed parents, ranked by WD
  - per-edge Wasserstein distance (causal effect size from interventional data)
  - the top enriched pathway terms, with likely-nonspecific terms FLAGGED
    (not removed) so the biologist can confirm/override
  - the top NON-FLAGGED significant term highlighted as the candidate annotation

The artifact denylist below auto-flags pathways that are large and gene-rich
enough to enrich on essential-gene composition rather than specific biology in
K562 (e.g. thyroid hormone signaling, viral-infection pathways, tissue-specific
differentiation). These are FLAGGED, not deleted -- a domain expert confirms.

USAGE:
    python build_handoff.py \
        --conv results/gnn_v4sweep_k562_pc7_seed4/subnet2_convergence.csv \
        --enrich results/gnn_v4sweep_k562_pc7_seed4/subnet2_enrichment_bg.csv \
        --model "GNN v4 (K562, pc7, seed4)" \
        --out results/gnn_v4sweep_k562_pc7_seed4/handoff_report.txt \
        --fdr 0.05
"""
import argparse
import re
import numpy as np
import pandas as pd

# --- Auto-flag denylist: pathways likely enriched by essential-gene composition
#     rather than specific K562 biology. FLAGGED for expert review, not deleted.
#     Edit/extend with your biologist; this is a defensible default, not final.
ARTIFACT_PATTERNS = [
    r"thyroid hormone",
    r"adipocyte",
    r"\bcoronavirus\b", r"covid",
    r"herpes simplex", r"epstein-barr", r"\bhiv\b",
    r"hepatitis", r"influenza", r"measles", r"\bsars\b",
    r"tuberculosis", r"salmonella", r"leishmania", r"toxoplasm", r"malaria",
    r"systemic lupus", r"rheumatoid",
    r"\bcancer\b",                 # generic "pathways in cancer" / "miRNAs in cancer"
    r"alcoholism", r"cocaine", r"amphetamine", r"morphine", r"nicotine",
    r"estrogen signaling", r"insulin signaling", r"glucagon",
    r"hippo signaling", r"\bampk\b",
    r"atherosclerosis", r"cardiomyopathy", r"diabetic",
    r"axon guidance", r"long-term potentiation", r"neuro",
    r"white adipocyte", r"osteoclast", r"melanogenesis",
]
ARTIFACT_RE = re.compile("|".join(ARTIFACT_PATTERNS), re.IGNORECASE)


def is_artifact(term):
    return bool(ARTIFACT_RE.search(term or ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", required=True)
    ap.add_argument("--enrich", required=True)
    ap.add_argument("--model", default="model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--top_terms", type=int, default=4,
                    help="how many top enrichment terms to show per subnetwork")
    args = ap.parse_args()

    conv = pd.read_csv(args.conv)
    enr = pd.read_csv(args.enrich)

    # conv anchor column is 'extended' (G+) or 'anchor' (G0); normalize to 'extended'
    if "extended" not in conv.columns and "anchor" in conv.columns:
        conv = conv.rename(columns={"anchor": "extended"})

    # rank subnetworks by median parent WD (causal strength), like the console preview
    med = conv.groupby("extended")["wd"].median().rename("median_wd")
    order = med.sort_values(ascending=False).index.tolist()

    # index enrichment by extended gene (enrichment CSV always uses 'extended')
    enr_by_ext = {ext: grp.sort_values("adj_p") for ext, grp in enr.groupby("extended")}

    lines = []
    W = lines.append
    W(f"# Convergence-subnetwork handoff report")
    W(f"# model: {args.model}")
    W(f"# subnetworks: {len(order)} | enrichment: KEGG_2021_Human + Reactome_2022, "
      f"G0-background, FDR<{args.fdr}")
    W(f"#")
    W(f"# Each subnetwork = an extended (anchor) gene + the perturbed genes that")
    W(f"# converge on it (prob>0.5 AND Wasserstein>=0.3). WD = effect size of")
    W(f"# perturbing the parent on the anchor's expression (interventional vs")
    W(f"# observational), a direct data-driven signal independent of any database.")
    W(f"#")
    W(f"# [FLAG] marks pathway terms auto-flagged as likely-nonspecific (large,")
    W(f"# gene-rich pathways that enrich on essential-gene composition). These are")
    W(f"# flagged for expert review, NOT removed. Confirm/override with domain knowledge.")
    W(f"#")

    # summary table first
    W("## SUMMARY (subnetworks ranked by median parent WD)\n")
    W(f"{'anchor':14s} {'n':>3s} {'medWD':>6s}  {'top non-flagged term':40s} {'adjP':>9s}")
    summary_rows = []
    for ext in order:
        grp = conv[conv.extended == ext]
        n = len(grp); mwd = grp["wd"].median()
        e = enr_by_ext.get(ext)
        top_clean, top_clean_p = "(none significant)", np.nan
        if e is not None:
            sig = e[e["adj_p"] < args.fdr]
            clean = sig[~sig["top_term"].apply(is_artifact)]
            if len(clean):
                r = clean.iloc[0]
                top_clean, top_clean_p = r["top_term"], r["adj_p"]
        W(f"{ext:14s} {n:3d} {mwd:6.2f}  {top_clean[:40]:40s} "
          f"{('%.1e'%top_clean_p) if not np.isnan(top_clean_p) else '   --   ':>9s}")
        summary_rows.append({"anchor": ext, "n_parents": n, "median_wd": mwd,
                             "top_nonflagged_term": top_clean, "adj_p": top_clean_p})

    # per-subnetwork detail
    W("\n\n## DETAIL\n")
    for ext in order:
        grp = conv[conv.extended == ext].sort_values("wd", ascending=False)
        n = len(grp); mwd = grp["wd"].median()
        W(f"\n{'='*70}")
        W(f"### {ext}   ({n} parents, median WD {mwd:.2f})")
        W(f"{'-'*70}")
        W(f"Gene set (for DAVID/STRING): {', '.join(sorted(set(grp['parent']) | {ext}))}")
        W(f"\nConverging edges (parent -> {ext}), by WD:")
        for _, r in grp.iterrows():
            W(f"   {r['parent']:12s} -> {ext:12s}  WD={r['wd']:.3f}  prob={r['prob']:.3f}")
        # enrichment
        e = enr_by_ext.get(ext)
        W(f"\nEnrichment (top {args.top_terms}, KEGG+Reactome, G0 background):")
        if e is None or len(e) == 0:
            W("   (no enrichment results -- library API may have failed; rerun)")
        else:
            shown = e.head(args.top_terms)
            for _, r in shown.iterrows():
                flag = " [FLAG: likely-nonspecific]" if is_artifact(r["top_term"]) else ""
                sigmark = "*" if r["adj_p"] < args.fdr else " "
                W(f"   {sigmark} {r['top_term'][:50]:50s} adjP={r['adj_p']:.2e} "
                  f"[{r['library'].split('_')[0]}]{flag}")
            # candidate annotation = top non-flagged significant
            sig = e[e["adj_p"] < args.fdr]
            clean = sig[~sig["top_term"].apply(is_artifact)]
            if len(clean):
                r = clean.iloc[0]
                W(f"\n   => CANDIDATE ANNOTATION: {r['top_term']} "
                  f"(adjP={r['adj_p']:.2e}, {r['overlap_n']} genes overlap)")
            elif len(sig):
                W(f"\n   => only flagged terms significant -- expert review needed")
            else:
                W(f"\n   => no significant enrichment at FDR<{args.fdr}")

    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    pd.DataFrame(summary_rows).to_csv(args.out.replace(".txt", "_summary.csv"), index=False)
    print(f"[saved] {args.out}")
    print(f"[saved] {args.out.replace('.txt', '_summary.csv')}")

    # console: how many have a clean candidate annotation
    n_clean = sum(1 for r in summary_rows if not (isinstance(r["adj_p"], float) and np.isnan(r["adj_p"])))
    print(f"\n[summary] {n_clean}/{len(order)} subnetworks have a non-flagged "
          f"significant pathway (candidate annotations for the case study)")
    print("[summary] top non-flagged candidates:")
    for r in sorted([r for r in summary_rows if not (isinstance(r['adj_p'],float) and np.isnan(r['adj_p']))],
                    key=lambda r: r["adj_p"])[:15]:
        print(f"   {r['anchor']:12s} {r['top_nonflagged_term'][:46]:46s} adjP={r['adj_p']:.1e}")


if __name__ == "__main__":
    main()