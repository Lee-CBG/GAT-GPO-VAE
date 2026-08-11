import sys, os, glob, json
from os.path import join, basename
import numpy as np, pandas as pd, anndata, scipy as sp, torch
sys.path.insert(0, '.')
from eval import get_ate_metrics, load_checkpoint
from gpo_vae.data.utils.anndata import align_adatas

RUN = sys.argv[1]
NP = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
best = [x for x in os.listdir(join(RUN, 'checkpoints')) if x.startswith('best')]
assert len(best) == 1, best
ckpt = join(RUN, 'checkpoints', best[0])
print('checkpoint:', ckpt, flush=True)

lm = load_checkpoint(ckpt, None)
dm = lm.get_data_module()
pred = lm.predictor

ref_all = dm.get_estimated_average_treatment_effects(method='perturbseq', qc_pass=False)
model_ate = pred.estimate_average_effects_data_module(
    data_module=dm, control_label=ref_all.uns['control'], method='perturbseq',
    n_particles=NP, condition_values=dict(library_size=10000*torch.ones((1,))),
    batch_size=128)

a = dm.adata
lab = a.obs['T'].astype(str).values
split = a.obs['split'].astype(str).values
X = a.X.toarray() if sp.sparse.issparse(a.X) else np.asarray(a.X)
m = split == 'test'
X = 1e4 * X / np.maximum(X.sum(1, keepdims=True), 1e-12)
X = np.log2(X + 1)
ctrl = X[m & (lab == 'non-targeting')].mean(0)
perts = [p for p in sorted(set(lab[m])) if p != 'non-targeting']
ref_test = anndata.AnnData(
    X=np.stack([X[m & (lab == p)].mean(0) - ctrl for p in perts]).astype(np.float64),
    obs=pd.DataFrame(index=perts), var=a.var.copy())

for name, ref in [('all-cells reference (published)', ref_all),
                  ('test-cells-only reference', ref_test)]:
    r, mo = align_adatas(ref, model_ate)
    mm = get_ate_metrics(r, mo)
    print('\n== %s   n_perts=%d' % (name, r.shape[0]))
    for k in ['ATE_pearsonr', 'ATE_r2', 'jaccard_sim_top50']:
        print('   %-20s %.4f' % (k, mm[k]))
