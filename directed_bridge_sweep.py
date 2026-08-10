"""
directed_bridge_sweep.py

Tests the GPO-VAE paper's LITERAL subnetwork definition: directed
G0 -> G+ -> G0 paths ("perturbed genes connected via at least one extended gene"),
across probability thresholds 0.5/0.6/0.7/0.8.

For each threshold we count directed 2-hop motifs (perturbed p -> extended e ->
perturbed q) under increasingly strict requirements, to show WHERE and WHETHER
the paper's bridges actually exist:

  (A) raw           : prob(p->e) > thr AND prob(e->q) > thr  (paper's loose rule)
  (B) + no-direct   : AND no direct p->q edge > thr          (paper's criterion ii)
  (C) + selective   : AND extended e is not a broadcast hub  (out-deg to G0 <= max_outdeg)
  (D) + WD both legs: AND WD(p->e) >= wd_min AND WD(e->q) >= wd_min  (real causal signal)

The headline result: (A)/(B) may be large (loose closure over a saturated block),
but (D) -- requiring real WD support on BOTH legs -- collapses to ~zero because the
G+->G0 second leg is saturated/non-selective. That is the evidence that the
directed-bridge definition does not hold up, justifying the convergence approach.

Also prints the two legs' density profiles per threshold to show the mechanism:
G0->G+ (first leg) thins fast as thr rises; G+->G0 (second leg) stays dense --
there is no threshold where both legs are simultaneously sparse-and-selective.

USAGE (repo root, eval env):
    CUDA_VISIBLE_DEVICES=1 WANDB_MODE=disabled python directed_bridge_sweep.py \
        --experiment_path results/gnn_v4sweep_k562_pc7_seed4 \
        --pert_gene_len 622 --devices 0 \
        --thresholds 0.5 0.6 0.7 0.8 \
        --wd_min 0.3 --max_outdeg 50 \
        --max_eval_motifs 4000
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
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8])
    ap.add_argument("--wd_min", type=float, default=0.3)
    ap.add_argument("--max_outdeg", type=int, default=50,
                    help="extended intermediate broadcast cap (G0 out-degree)")
    ap.add_argument("--p_value_threshold", type=float, default=0.05)
    ap.add_argument("--max_eval_motifs", type=int, default=4000,
                    help="cap candidate motifs WD-evaluated per threshold (runtime bound)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    PERT = args.pert_gene_len
    out = args.out or join(args.experiment_path, "directed_bridge_sweep.csv")

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

    grn = pd.read_csv(join(args.experiment_path, "grn.csv"), index_col=0)
    genes = grn.index.tolist()
    P = grn.values.copy()
    np.fill_diagonal(P, 0.0)
    n = len(genes)

    _wd = {}
    def wd(a, b):
        k = (a, b)
        if k in _wd:
            return _wd[k]
        if a not in g2i or b not in g2idx:
            _wd[k] = None; return None
        try:
            o = predictor.get_observational(b); iv = predictor.get_interventional(b, a)
        except KeyError:
            _wd[k] = None; return None
        v = None if (len(o) == 0 or len(iv) == 0) else \
            float(scipy.stats.wasserstein_distance(o, iv))
        _wd[k] = v
        return v

    g0_gp = P[:PERT, PERT:]            # G0 -> G+
    gp_g0 = P[PERT:, :PERT]            # G+ -> G0
    g0_g0 = P[:PERT, :PERT]            # G0 -> G0 (for the no-direct-edge filter)

    rows = []
    print(f"\n{'thr':>4s} | {'G0->G+ edges':>12s} {'G+->G0 edges':>12s} | "
          f"{'(A)raw':>9s} {'(B)+noDir':>9s} {'(C)+selec':>9s} {'(D)+WDboth':>10s}")
    print("-" * 80)

    for thr in args.thresholds:
        leg1 = g0_gp > thr            # [PERT, n-PERT]
        leg2 = gp_g0 > thr            # [n-PERT, PERT]
        direct = g0_g0 > thr          # [PERT, PERT]
        ext_outdeg = leg2.sum(1)      # per extended gene, # G0 children

        n_leg1 = int(leg1.sum())
        n_leg2 = int(leg2.sum())

        # enumerate candidate 2-hop motifs p -> e -> q
        countA = countB = countC = 0
        candidates_for_D = []         # (p_idx, e_global, q_idx) passing C
        p_idx, e_loc = np.nonzero(leg1)
        # group children by extended gene for speed
        for a, el in zip(p_idx, e_loc):
            children = np.nonzero(leg2[el])[0]
            if len(children) == 0:
                continue
            selective = ext_outdeg[el] <= args.max_outdeg
            for q in children:
                if q == a:
                    continue
                countA += 1
                if direct[a, q]:
                    continue
                countB += 1
                if not selective:
                    continue
                countC += 1
                candidates_for_D.append((a, PERT + el, q))

        # (D) WD on both legs -- cap evaluation for runtime
        countD = 0
        evalset = candidates_for_D
        capped = False
        if len(evalset) > args.max_eval_motifs:
            capped = True
            # random sample to estimate the rate, then scale
            idx = np.random.default_rng(0).choice(len(evalset), args.max_eval_motifs, replace=False)
            evalset = [candidates_for_D[i] for i in idx]
        passD = 0
        for (a, eg, q) in evalset:
            w1 = wd(genes[a], genes[eg])
            if w1 is None or w1 < args.wd_min:
                continue
            w2 = wd(genes[eg], genes[q])
            if w2 is None or w2 < args.wd_min:
                continue
            passD += 1
        if capped:
            countD = f"~{int(passD * len(candidates_for_D) / len(evalset))}*"
        else:
            countD = str(passD)

        print(f"{thr:4.2f} | {n_leg1:12d} {n_leg2:12d} | "
              f"{countA:9d} {countB:9d} {countC:9d} {str(countD):>10s}")
        rows.append({"threshold": thr, "leg1_edges": n_leg1, "leg2_edges": n_leg2,
                     "A_raw": countA, "B_no_direct": countB, "C_selective": countC,
                     "D_wd_both": countD, "D_capped": capped})

    pd.DataFrame(rows).to_csv(out, index=False)
    print("-" * 80)
    print("* (D) capped: rate estimated from a sample of candidates, scaled up.")
    print(f"\n[saved] {out}")
    print("\nInterpretation: if (D) collapses toward 0 across thresholds while (A)/(B)")
    print("stay large, the paper's directed bridges exist only via loose closure over")
    print("the saturated G+->G0 leg, NOT as WD-supported directed paths. Note also how")
    print("leg1 (G0->G+) thins as thr rises while leg2 (G+->G0) stays dense -- there is")
    print("no threshold where both legs are simultaneously sparse and selective.")


if __name__ == "__main__":
    main()