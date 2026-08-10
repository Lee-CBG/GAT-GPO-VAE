import os, re, glob
import pandas as pd, numpy as np

METRICS={'ATE_pearsonr-all':'ATE-rho','ATE_r2-all':'ATE-R2','jaccard_sim_top50-all':'Jac50',
 'positive_mean_wasserstein__g0':'muWD','false_omission_rate__g0':'FOR','num_edges':'edges'}
ALT={'positive_mean_wasserstein__g0':'positive_mean_wasserstein','false_omission_rate__g0':'false_omission_rate'}

def canon_csv(cfg):
    cands=[]
    for f in glob.glob(f"results/*{cfg}*/test_metrics.csv"):
        d=os.path.dirname(f)
        if re.search(rf"/{re.escape(cfg)}(-[0-9]+)?$",d):
            ck=glob.glob(os.path.join(d,"checkpoints","best*.ckpt"))
            ep=max([int(re.search(r'epoch=(\d+)',c).group(1)) for c in ck],default=-1)
            cands.append((ep,f))
    return max(cands)[1] if cands else None

def load(cfg):
    f=canon_csv(cfg)
    if not f: return None
    df=pd.read_csv(f,index_col=0); col=df.columns[0]; out={}
    for k,n in METRICS.items():
        key=k if k in df.index else ALT.get(k)
        out[n]=float(df.loc[key,col]) if key in df.index else np.nan
    return out

# baseline configs (2/4/256/0.0) = the winning-pc 5-seed runs; use seeds 0-2 for fair 3-seed compare
BASE={'rpe1':[f'gnn_v4sweep_rpe1_pc15_seed{s}' for s in range(3)],
      'k562':[f'gnn_v4sweep_k562_pc7_seed{s}' for s in range(3)],
      'adamson':[f'gnn_v4sweep_adamson_pc0p5_seed{s}' for s in range(3)]}
# knob -> tags (baseline is the implicit middle)
KNOBS={'layers':[('1','L1'),('2(base)','BASE'),('3','L3')],
       'heads':[('2','H2'),('4(base)','BASE'),('8','H8')],
       'd_hidden':[('128','D128'),('256(base)','BASE'),('512','D512')],
       'dropout':[('0.0(base)','BASE'),('0.1','drop0p1'),('0.3','drop0p3')]}
order=['ATE-rho','ATE-R2','Jac50','muWD','FOR','edges']

def agg(cfgs):
    rows=[load(c) for c in cfgs]; rows=[r for r in rows if r]
    if not rows: return None,0
    out={}
    for m in order:
        a=np.array([r[m] for r in rows],float); out[m]=(np.nanmean(a),np.nanstd(a))
    return out,len(rows)

for ds in ['rpe1','k562','adamson']:
    print(f"\n{'#'*84}\n# {ds.upper()}  (baseline = 2 layers / 4 heads / 256 hidden / 0.0 dropout)\n{'#'*84}")
    for knob,vals in KNOBS.items():
        print(f"\n  --- {knob} ---")
        for label,tag in vals:
            cfgs = BASE[ds] if tag=='BASE' else [f'gnn_arch_{ds}_{tag}_seed{s}' for s in range(3)]
            res,n=agg(cfgs)
            if not res: print(f"    {label:<12} [no data]"); continue
            line=f"    {label:<12} n={n}  "
            for m in order:
                mu,sd=res[m]
                line+=(f"{m}={mu:6.0f} " if m=='edges' else f"{m}={mu:.3f} ")
            print(line)
