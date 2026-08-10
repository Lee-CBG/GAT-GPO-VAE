"""
enrich_subnets_bg.py

Functional enrichment of convergence subnetworks with the CORRECT background:
the K562 perturbed-gene universe (G0 = first --pert_gene_len genes), not the whole
genome. Uses the Speedrichr API (maayanlab.cloud/speedrichr), which supports custom
backgrounds -- the classic /Enrichr/ endpoints do not.

Why this matters: the perturbed-gene pool is overwhelmingly essential genes
(ribosome/proteasome/splicing/translation). Against a whole-genome background,
ANY subset enriches for those terms -> everything looks significant (the inflated
97/97). Against the G0 background, a subnetwork is significant only if it captures
a pathway SPECIFIC beyond the baseline essential-gene composition. That is the
publishable test.

Speedrichr flow:
  POST /api/addbackground   {background}            -> {backgroundid}
  POST /api/addList         {list, description}      -> {userListId}
  POST /api/backgroundenrich{userListId, backgroundid, backgroundType=library}
                                                     -> {library: [rows]}
  row: [rank, term, p, oddsratio, combined, overlap_genes, adj_p, ...]

USAGE (repo root):
    python enrich_subnets_bg.py \
        --conv results/gpo_vae_replogle_seed0/subnet2_convergence.csv \
        --grn  results/gpo_vae_replogle_seed0/grn.csv \
        --pert_gene_len 622 \
        --out  results/gpo_vae_replogle_seed0/subnet2_enrichment_bg.csv \
        --fdr 0.05
"""
import argparse
import json
import time
import urllib.request
import urllib.parse

import numpy as np
import pandas as pd

SPEED = "https://maayanlab.cloud/speedrichr/api"
# GO_Biological_Process libraries (2021 AND 2023) return HTTP 500 from speedrichr's
# backgroundenrich endpoint whenever the gene list contains a symbol GO doesn't
# recognize (lncRNAs / antisense / novel ORFs -- common as extended-gene anchors,
# e.g. APTR, *-AS1, AC######.#). KEGG and Reactome handle unrecognized genes
# gracefully. So default to KEGG + Reactome for a clean, fully-tested denominator.
# (Pass --libraries with a GO build explicitly if you want to attempt it.)
DEFAULT_LIBS = ["KEGG_2021_Human", "Reactome_2022"]


def _post_multipart(url, fields, retries=3):
    boundary = "----bx"
    body = ""
    for k, v in fields.items():
        body += (f"--{boundary}\r\n"
                 f'Content-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n')
    body += f"--{boundary}--\r\n"
    req = urllib.request.Request(
        url, data=body.encode(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    for attempt in range(retries):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=30).read().decode())
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def _post_json(url, payload, retries=3):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    for attempt in range(retries):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=30).read().decode())
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def add_background(genes):
    return _post_multipart(f"{SPEED}/addbackground",
                           {"background": "\n".join(genes)})["backgroundid"]


def add_list(genes):
    return _post_multipart(f"{SPEED}/addList",
                           {"list": "\n".join(genes), "description": "subnet"})["userListId"]


def background_enrich(user_list_id, background_id, library):
    # speedrichr's backgroundenrich requires urlencoded form data; JSON 500s.
    payload = urllib.parse.urlencode({
        "userListId": int(user_list_id),
        "backgroundid": background_id,
        "backgroundType": library,
    }).encode()
    req = urllib.request.Request(
        f"{SPEED}/backgroundenrich", data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"})
    # speedrichr's GO library intermittently 500s under load; retry hard with
    # exponential backoff. Raise on final failure so the caller records the
    # library as UNTESTED (not as "non-significant").
    last = None
    for attempt in range(6):
        try:
            data = json.loads(urllib.request.urlopen(req, timeout=60).read().decode())
            return data.get(library, [])
        except Exception as e:
            last = e
            time.sleep(min(2 ** attempt, 20))  # 1,2,4,8,16,20s
    raise last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", required=True)
    ap.add_argument("--grn", required=True)
    ap.add_argument("--pert_gene_len", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--libraries", nargs="+", default=DEFAULT_LIBS)
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--min_genes", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=0.4)
    args = ap.parse_args()

    # background = G0 perturbed-gene universe
    g0 = pd.read_csv(args.grn, index_col=0).index.tolist()[:args.pert_gene_len]
    print(f"[bg] registering G0 background ({len(g0)} genes)")
    bg_id = add_background(g0)
    print(f"[bg] backgroundid={bg_id}")

    conv = pd.read_csv(args.conv)
    # anchor column is 'extended' (G+ convergence) or 'anchor' (G0 convergence)
    anchor_col = "extended" if "extended" in conv.columns else "anchor"
    subnets = {}
    for ext, grp in conv.groupby(anchor_col):
        genes = sorted(set(grp["parent"]) | {ext})
        if len(genes) >= args.min_genes:
            subnets[ext] = genes
    print(f"[enrich] {len(subnets)} subnetworks (background-corrected, anchor='{anchor_col}')")

    rows = []
    coverage = {}   # ext -> (n_libs_ok, n_libs_failed)
    for i, (ext, genes) in enumerate(subnets.items()):
        try:
            uid = add_list(genes)
        except Exception as e:
            print(f"  [{ext}] addList failed: {e}")
            coverage[ext] = (0, len(args.libraries))
            continue
        best = None
        n_ok, n_fail = 0, 0
        for lib in args.libraries:
            time.sleep(args.sleep)
            try:
                res = background_enrich(uid, bg_id, lib)
                n_ok += 1
            except Exception as e:
                n_fail += 1
                print(f"  [{ext}/{lib}] enrich failed after retries: {str(e)[:60]}")
                continue
            if not res:
                continue
            top = min(res, key=lambda r: r[6])      # smallest adj p
            term, p, adj_p, overlap = top[1], top[2], top[6], top[5]
            rows.append({
                "model_seed": args.conv.split("/")[-2],
                "extended": ext, "n_genes": len(genes), "library": lib,
                "top_term": term, "adj_p": adj_p, "p": p,
                "overlap_n": len(overlap), "overlap_genes": ";".join(overlap),
            })
            if best is None or adj_p < best[1]:
                best = (term, adj_p, lib)
        coverage[ext] = (n_ok, n_fail)
        flag = "***" if (best and best[1] < args.fdr) else "   "
        if best:
            print(f"  {flag} {ext:12s} (n={len(genes):2d})  {best[0][:48]:48s} "
                  f"adjP={best[1]:.2e} [{best[2].split('_')[0]}]")
        else:
            print(f"      {ext:12s} (n={len(genes):2d})  -- no enrichment --")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    # coverage accounting: a subnetwork is FULLY TESTED only if all libraries
    # returned (n_fail == 0). Otherwise it is UNTESTED and excluded from the
    # significance denominator so API failures don't masquerade as negatives.
    n_libs = len(args.libraries)
    fully_tested = {e for e, (ok, fail) in coverage.items() if fail == 0}
    partial = {e for e, (ok, fail) in coverage.items() if 0 < ok and fail > 0}
    untested = {e for e, (ok, fail) in coverage.items() if ok == 0}
    print(f"\n[coverage] {len(fully_tested)} fully tested, {len(partial)} partial, "
          f"{len(untested)} fully failed (of {len(subnets)} subnetworks)")

    if len(df):
        best_per = df.loc[df.groupby("extended")["adj_p"].idxmin()]
        # CLEAN denominator: significance only among fully-tested subnetworks
        bp_full = best_per[best_per["extended"].isin(fully_tested)]
        n_sig_full = int((bp_full["adj_p"] < args.fdr).sum())
        print(f"[summary] CLEAN: {n_sig_full}/{len(fully_tested)} fully-tested "
              f"subnetworks significant at FDR<{args.fdr}")
        # also report sig among ALL that got at least one library (lower bound)
        n_sig_any = int((best_per["adj_p"] < args.fdr).sum())
        print(f"[summary] any-library lower bound: {n_sig_any}/{len(subnets)} significant")

        sigrows = best_per[best_per["adj_p"] < args.fdr]
        print("\n[summary] distinct significant terms (best library per subnetwork):")
        vc = sigrows["top_term"].value_counts()
        for term, c in vc.items():
            print(f"   {c:2d}x  {term[:62]}")
        print("\n[summary] top significant subnetworks:")
        for _, r in sigrows.sort_values("adj_p").head(25).iterrows():
            tested = "OK " if r["extended"] in fully_tested else "PART"
            print(f"   [{tested}] {r['extended']:12s}  {r['top_term'][:48]:48s} adjP={r['adj_p']:.2e}")

        if partial or untested:
            print(f"\n[note] {len(partial)} subnetworks had >=1 library fail "
                  f"(GO flakiness); rerun to fully test: "
                  f"{sorted(partial | untested)[:15]}{'...' if len(partial|untested)>15 else ''}")
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()