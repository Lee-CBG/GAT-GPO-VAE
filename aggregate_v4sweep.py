import os, re, glob
import pandas as pd, numpy as np

METRICS = {
    'ATE_pearsonr-all':              'ATE-rho',
    'ATE_r2-all':                    'ATE-R2',
    'jaccard_sim_top50-all':         'Jaccard50',
    'positive_mean_wasserstein__g0': 'muWD_g0',
    'false_omission_rate__g0':       'FOR_g0',
    'num_edges':                     'edges',
}
ALT = {'positive_mean_wasserstein__g0':'positive_mean_wasserstein',
       'false_omission_rate__g0':'false_omission_rate'}

def canon_csv(cfg):
    cands=[]
    for f in glob.glob(f"results/*{cfg}*/test_metrics.csv"):
        d=os.path.dirname(f)
        if re.search(rf"/{re.escape(cfg)}(-[0-9]+)?$", d):
            ck=glob.glob(os.path.join(d,"checkpoints","best*.ckpt"))
            ep=max([int(re.search(r'epoch=(\d+)',c).group(1)) for c in ck], default=-1)
            cands.append((ep,f))
    return max(cands)[1] if cands else None

def load(cfg):
    f=canon_csv(cfg)
    if not f: return None
    df=pd.read_csv(f,index_col=0)
    col=df.columns[0]
    out={}
    for k,name in METRICS.items():
        key=k if k in df.index else ALT.get(k)
        out[name]=float(df.loc[key,col]) if key in df.index else np.nan
    return out

configs=sorted(os.path.basename(p)[:-5] for p in glob.glob("demo/gnn_v4sweep_*.yaml"))
rows=[]
for cfg in configs:
    m=re.match(r"gnn_v4sweep_(\w+?)_pc([0-9p]+)_seed(\d+)",cfg)
    ds,pc,seed=m.group(1),m.group(2),int(m.group(3))
    r=load(cfg)
    if r is None: print(f"  [missing] {cfg}"); continue
    r.update(dict(dataset=ds,pc=pc.replace('p','.'),seed=seed))
    rows.append(r)

df=pd.DataFrame(rows)
order=['ATE-rho','ATE-R2','Jaccard50','muWD_g0','FOR_g0','edges']
paper={'rpe1':(0.651,0.402,0.216,0.280,0.042,2211),
       'k562':(0.766,0.570,0.322,0.248,0.022,2888),
       'adamson':(0.864,0.731,0.414,0.159,0.014,11)}
for ds in ['rpe1','k562','adamson']:
    sub=df[df.dataset==ds]
    if sub.empty: continue
    print(f"\n{'='*78}\n{ds.upper()}   (paper: ATE-rho={paper[ds][0]} R2={paper[ds][1]} Jac={paper[ds][2]} muWD={paper[ds][3]} FOR={paper[ds][4]} edges={paper[ds][5]})\n{'='*78}")
    for pc,grp in sorted(sub.groupby('pc'), key=lambda x: float(x[0])):
        n=len(grp); line=f"  pc={pc:<5} n={n}  "
        for col in order:
            mu=grp[col].mean(); sd=grp[col].std(ddof=0)
            if col=='edges': line+=f"{col}={mu:6.0f}±{sd:<4.0f} "
            else:           line+=f"{col}={mu:.3f}±{sd:.3f} "
        print(line)
