"""
enrich_subnets.py

Functional enrichment of convergence subnetworks (subnetworks_v2.py output) using
the Enrichr REST API directly (no gseapy needed). For each subnetwork's gene set,
reports the top enriched term per library with its Benjamini-Hochberg adjusted
p-value, and flags subnetworks significant at FDR < --fdr.

This turns the analysis from "here are 4 hand-picked modules" into "here are ALL N
convergence subnetworks, M of which are significantly enriched for a named pathway."

Enrichr libraries queried (override with --libraries):
    GO_Biological_Process_2021
    KEGG_2021_Human
    Reactome_2022

USAGE (repo root; needs outbound HTTPS to maayanlab.cloud, already verified):
    python enrich_subnets.py \
        --conv results/gpo_vae_replogle_seed0/subnet2_convergence.csv \
        --out  results/gpo_vae_replogle_seed0/subnet2_enrichment.csv \
        --fdr 0.05

Enrichr API:
  POST /Enrichr/addList   (genes) -> userListId
  GET  /Enrichr/enrich?userListId=..&backgroundType=LIBRARY -> results
Result row format (per Enrichr): [rank, term, p, z, combined, genes, adj_p, ...]
"""
import argparse
import json
import time
import urllib.request
import urllib.parse

import numpy as np
import pandas as pd

ENRICHR = "https://maayanlab.cloud/Enrichr"
DEFAULT_LIBS = ["GO_Biological_Process_2021", "KEGG_2021_Human", "Reactome_2022"]


def enrichr_add_list(genes, retries=3):
    """Submit a gene list, return userListId."""
    genes_str = "\n".join(genes)
    # multipart/form-data with a single field 'list'
    boundary = "----enrichrboundary1234567890"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="list"\r\n\r\n'
        f"{genes_str}\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    req = urllib.request.Request(
        f"{ENRICHR}/addList", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    for attempt in range(retries):
        try:
            r = urllib.request.urlopen(req, timeout=20)
            return json.loads(r.read().decode())["userListId"]
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def enrichr_enrich(user_list_id, library, retries=3):
    """Fetch enrichment results for a library; return list of result rows."""
    url = f"{ENRICHR}/enrich?userListId={user_list_id}&backgroundType={urllib.parse.quote(library)}"
    for attempt in range(retries):
        try:
            r = urllib.request.urlopen(url, timeout=20)
            data = json.loads(r.read().decode())
            return data.get(library, [])
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", required=True, help="subnet2_convergence.csv path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--libraries", nargs="+", default=DEFAULT_LIBS)
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--min_genes", type=int, default=3,
                    help="skip subnetworks with fewer than this many genes")
    ap.add_argument("--sleep", type=float, default=0.4,
                    help="pause between Enrichr calls (politeness/rate-limit)")
    args = ap.parse_args()

    conv = pd.read_csv(args.conv)
    if len(conv) == 0:
        print("[warn] empty convergence file; nothing to enrich")
        return

    # build one gene set per extended gene: {extended} U parents
    subnets = {}
    for ext, grp in conv.groupby("extended"):
        genes = sorted(set(grp["parent"]) | {ext})
        if len(genes) >= args.min_genes:
            subnets[ext] = genes
    print(f"[enrich] {len(subnets)} subnetworks from {args.conv}")

    rows = []
    for i, (ext, genes) in enumerate(subnets.items()):
        try:
            uid = enrichr_add_list(genes)
        except Exception as e:
            print(f"  [{ext}] addList failed: {e}")
            continue
        best_overall = None
        for lib in args.libraries:
            time.sleep(args.sleep)
            try:
                res = enrichr_enrich(uid, lib)
            except Exception as e:
                print(f"  [{ext}/{lib}] enrich failed: {e}")
                continue
            if not res:
                continue
            # res rows: [rank, term, p, z, combined, overlap_genes, adj_p, ...]
            top = min(res, key=lambda r: r[6])  # smallest adjusted p
            term, adj_p, p, overlap_genes = top[1], top[6], top[2], top[5]
            rows.append({
                "model_seed": args.conv.split("/")[-2],
                "extended": ext, "n_genes": len(genes), "library": lib,
                "top_term": term, "adj_p": adj_p, "p": p,
                "overlap_n": len(overlap_genes),
                "overlap_genes": ";".join(overlap_genes),
            })
            if best_overall is None or adj_p < best_overall[1]:
                best_overall = (term, adj_p, lib, len(overlap_genes))
        sig = "***" if (best_overall and best_overall[1] < args.fdr) else "   "
        if best_overall:
            print(f"  {sig} {ext:12s} (n={len(genes):2d})  {best_overall[0][:50]:50s} "
                  f"adjP={best_overall[1]:.2e} [{best_overall[2].split('_')[0]}]")
        else:
            print(f"      {ext:12s} (n={len(genes):2d})  -- no enrichment --")
        if (i + 1) % 10 == 0:
            print(f"  ...{i+1}/{len(subnets)} subnetworks done")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    # summary: how many subnetworks have ANY library term below FDR
    if len(df):
        best_per = df.loc[df.groupby("extended")["adj_p"].idxmin()]
        n_sig = int((best_per["adj_p"] < args.fdr).sum())
        print(f"\n[summary] {n_sig}/{len(subnets)} subnetworks significant at FDR<{args.fdr}")
        print("[summary] top significant subnetworks:")
        show = best_per[best_per["adj_p"] < args.fdr].sort_values("adj_p").head(20)
        for _, r in show.iterrows():
            print(f"   {r['extended']:12s}  {r['top_term'][:55]:55s} adjP={r['adj_p']:.2e}")
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()