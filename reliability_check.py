import sys, glob
import numpy as np, pandas as pd, anndata, scipy as sp
sys.path.insert(0, '.')
from eval import get_ate_metrics

DS = sys.argv[1] if len(sys.argv) > 1 else 'rpe1'
a = anndata.read_h5ad(glob.glob('datasets/*' + DS + '*.h5ad')[0])
lab_all = a.obs['T'].astype(str).values
split = a.obs['split'].astype(str).values
X_all = a.X.toarray() if sp.sparse.issparse(a.X) else np.asarray(a.X)

def ate(mask):
    lab = lab_all[mask]; X = X_all[mask]
    X = 1e4 * X / np.maximum(X.sum(1, keepdims=True), 1e-12)
    X = np.log2(X + 1)
    ctrl = X[lab == 'non-targeting'].mean(0)
    perts = sorted(set(lab) - {'non-targeting'})
    return perts, np.stack([X[lab == p].mean(0) - ctrl for p in perts])

def score(pA, A, pB, B, name):
    common = [p for p in pA if p in set(pB)]
    A2 = A[[pA.index(p) for p in common]]
    B2 = B[[pB.index(p) for p in common]]
    idx = pd.DataFrame(index=common)
    m = get_ate_metrics(anndata.AnnData(X=A2.astype(np.float64), obs=idx),
                        anndata.AnnData(X=B2.astype(np.float64), obs=idx))
    print('%-38s n=%3d  rho=%.4f  r2=%7.4f  jac50=%.4f'
          % (name, len(common), m['ATE_pearsonr'], m['ATE_r2'], m['jaccard_sim_top50']))

rng = np.random.default_rng(0)
tr = np.where(split == 'train')[0]
half = rng.permutation(tr)
hA = np.zeros(len(lab_all), bool); hA[half[:len(tr)//2]] = True
hB = np.zeros(len(lab_all), bool); hB[half[len(tr)//2:]] = True

pA, A = ate(hA); pB, B = ate(hB)
pT, T = ate(split == 'test')
pV, V = ate(split == 'val')
pTr, Tr = ate(split == 'train')

print('cells: train %d  val %d  test %d' % ((split=='train').sum(), (split=='val').sum(), (split=='test').sum()))
print('')
score(pA, A, pB, B,  'train-half-A vs train-half-B')
score(pT, T, pV, V,  'test vs val  (both held out)')
score(pT, T, pTr, Tr,'test vs train  (= the baseline)')
