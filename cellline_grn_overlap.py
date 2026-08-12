"""
R5-W2: cross-cell-line GRN comparison.
RPE1 vs K562 learned GRNs restricted to the genes PERTURBED in both lines,
tested against a degree-preserving null. Also reports the same statistic on
the wider universe including extended (never-perturbed) genes, as a contrast.
Usage: python cellline_grn_overlap.py
"""
import numpy as np, pandas as pd, anndata
from os.path import join

RPE1 = ['results/gnn_v4sweep_rpe1_pc15_seed%d' % i for i in (0,1,2)] + \
       ['results/gnn_v4_rpe1_pc15_seed%d' % i for i in (3,4)]
K562 = ['results/gnn_v4sweep_k562_pc7_seed%d' % i for i in range(5)]
DATA = {'rpe1': './datasets/rpe1_qc_deg_matched_ctrl_idx_all_ot.h5ad',
        'k562': './datasets/k562_qc_deg_matched_ctrl_idx_all_ot.h5ad'}
NPERM, THR = 2000, 0.5
RNG = np.random.default_rng(0)

pert = {}
for k, p in DATA.items():
    a = anndata.read_h5ad(p, backed='r')
    pert[k] = set(map(str, a.obs['T'].unique())) - {'non-targeting'}

grn = {d: pd.read_csv(join(d, 'grn.csv'), index_col=0) for d in RPE1 + K562}
cols_r, cols_k = set(grn[RPE1[0]].columns), set(grn[K562[0]].columns)
U_pert = sorted(pert['rpe1'] & pert['k562'] & cols_r & cols_k)
U_all  = sorted(cols_r & cols_k)
print('shared perturbed genes: %d   shared any-type genes: %d' % (len(U_pert), len(U_all)))

def adj(d, genes):
    a = (grn[d].loc[genes, genes].values > THR)
    np.fill_diagonal(a, False)
    return a

def swap_null(A, B, n=NPERM):
    """Degree-preserving double-edge swaps on A; B fixed."""
    obs = int((A & B).sum())
    src, dst = np.nonzero(A)
    E = len(src)
    if E == 0:
        return obs, 0.0, 1.0, E
    hits = np.empty(n, dtype=int)
    for t in range(n):
        s, d = src.copy(), dst.copy()
        for _ in range(5 * E):
            i, j = RNG.integers(0, E, 2)
            if s[i] == d[j] or s[j] == d[i]:
                continue
            d[i], d[j] = d[j], d[i]
        P = np.zeros_like(A)
        P[s, d] = True
        np.fill_diagonal(P, False)
        hits[t] = int((P & B).sum())
    p = (np.sum(hits >= obs) + 1) / (n + 1)
    return obs, float(hits.mean()), float(p), E

for label, U in [('shared PERTURBED genes', U_pert), ('incl. extended genes', U_all)]:
    rows = []
    for i, (rd, kd) in enumerate(zip(RPE1, K562)):
        A, B = adj(rd, U), adj(kd, U)
        obs, exp, p, Ea = swap_null(A, B)
        rows.append(dict(seed=i, rpe1_edges=Ea, k562_edges=int(B.sum()), overlap=obs,
                         expected=round(exp, 1), fold=round(obs / max(exp, 1e-9), 2),
                         p=round(p, 5),
                         jaccard=round(obs / max(Ea + int(B.sum()) - obs, 1), 4)))
    df = pd.DataFrame(rows)
    print('\n=== %s (n=%d) ===' % (label, len(U)))
    print(df.to_string(index=False))
    print('mean fold %.2f   median p %.4g' % (df.fold.mean(), df.p.median()))
    df.to_csv('cellline_overlap_%s.csv' % ('pert' if U is U_pert else 'all'), index=False)

# reproducible line-specific edges, perturbed universe only
rs = [set(zip(*np.nonzero(adj(d, U_pert)))) for d in RPE1]
ks = [set(zip(*np.nonzero(adj(d, U_pert)))) for d in K562]
def stable(sets, k=4):
    c = {}
    for s in sets:
        for e in s: c[e] = c.get(e, 0) + 1
    return {e for e, n in c.items() if n >= k}
g = np.array(U_pert)
r_only = stable(rs) - set().union(*ks)
k_only = stable(ks) - set().union(*rs)
print('\nedges in >=4/5 seeds of one line and 0/5 seeds of the other:')
print('  RPE1-specific %d   K562-specific %d   stable in both %d'
      % (len(r_only), len(k_only), len(stable(rs) & stable(ks))))
print('  RPE1 e.g.:', [f'{g[i]}->{g[j]}' for i, j in sorted(r_only)[:8]])
print('  K562 e.g.:', [f'{g[i]}->{g[j]}' for i, j in sorted(k_only)[:8]])
