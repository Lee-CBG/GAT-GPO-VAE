import sys, glob, json
import numpy as np, pandas as pd, anndata, scipy as sp
sys.path.insert(0, '.')
from eval import get_ate_metrics

DS = sys.argv[1] if len(sys.argv) > 1 else 'rpe1'
path = glob.glob('datasets/*' + DS + '*.h5ad')[0]
print('dataset:', path, flush=True)
a = anndata.read_h5ad(path)

def ate(adata, mask):
    sub = adata[mask]
    lab = sub.obs['T'].astype(str).values
    X = sub.X.toarray() if sp.sparse.issparse(sub.X) else np.asarray(sub.X)
    X = 1e4 * X / np.maximum(X.sum(1, keepdims=True), 1e-12)
    X = np.log2(X + 1)
    ctrl = X[lab == 'non-targeting'].mean(0)
    perts = sorted(set(lab) - {'non-targeting'})
    return perts, np.stack([X[lab == p].mean(0) - ctrl for p in perts])

split = a.obs['split'].astype(str).values
perts_te, ATE_te = ate(a, split == 'test')
perts_tr, ATE_tr = ate(a, split == 'train')

common = [p for p in perts_te if p in set(perts_tr)]
ref = ATE_te[[perts_te.index(p) for p in common]]
b_pert = ATE_tr[[perts_tr.index(p) for p in common]]
b_glob = np.repeat(ATE_tr.mean(0, keepdims=True), len(common), axis=0)
print('perturbations scored:', len(common), ' genes:', ref.shape[1], flush=True)

def wrap(M):
    return anndata.AnnData(X=np.asarray(M, dtype=np.float64),
                           obs=pd.DataFrame(index=common))

R = wrap(ref)
out = {}
out['train_split_mean_per_perturbation'] = get_ate_metrics(R, wrap(b_pert))
out['global_train_mean'] = get_ate_metrics(R, wrap(b_glob))
for k in out:
    print('')
    print('==', k)
    for m in ['ATE_pearsonr', 'ATE_r2', 'jaccard_sim_top50']:
        print('   %-22s %.4f' % (m, out[k][m]))

json.dump({k: {m: float(x) for m, x in v.items()} for k, v in out.items()},
          open('linear_baselines_' + DS + '.json', 'w'), indent=2)
print('')
print('wrote linear_baselines_' + DS + '.json')
