import os, re, glob
import pandas as pd, numpy as np

METRICS = {
    'ATE_pearsonr-all':'ATE-rho','ATE_r2-all':'ATE-R2','jaccard_sim_top50-all':'Jaccard50',
    'positive_mean_wasserstein__g0':'muWD_g0','false_omission_rate__g0':'FOR_g0','num_edges':'edges',
}
ALT={'positive_mean_wasserstein__g0':'positive_mean_wasserstein','false_omission_rate__g0':'false_omission_rate'}

def canon_csv(cfg):
    cands=[]
    for f in glob.glob(f"results/*{cfg}*/test_metrics.csv"):
        d=os.path.dirname(f)
        if re.search(rf"/{re.escape(cfg)}(-[0-9]+)?$", d):
            ck=glob.glob(os.path.join(d,"checkpoints","best*.ckpt"))
            ep=max([int(re.search(r'epoch=(\d+)',c).group(1)) for c in ck],default=-1)
            cands.append((ep,f))
    return max(cands)[1] if cands else None

def load(cfg):
    f=canon_csv(cfg)
    if not f: return None
    df=pd.read_csv(f,index_col=0); col=df.columns[0]; out={}
    for k,name in METRICS.items():
        key=k if k in df.index else ALT.get(k)
        out[name]=float(df.loc[key,col]) if key in df.index else np.nan
    return out

# 9 settings: dataset -> pc -> list of 5 config names (seeds 0-4), mixing naming schemes
SETTINGS={
 'rpe1':{
   '12':[f'gnn_v4sweep_rpe1_pc12_seed{s}' for s in range(5)],
   '15':[f'gnn_v4sweep_rpe1_pc15_seed{s}' for s in range(3)]+[f'gnn_v4_rpe1_pc15_seed{s}' for s in (3,4)],
   '18':[f'gnn_v4sweep_rpe1_pc18_seed{s}' for s in range(5)],
 },
 'k562':{
   '4':[f'gnn_v4sweep_k562_pc4_seed{s}' for s in range(5)],
   '5':[f'gnn_v4sweep_k562_pc5_seed{s}' for s in range(3)]+[f'gnn_v4_k562_pc5_seed{s}' for s in (3,4)],
   '7':[f'gnn_v4sweep_k562_pc7_seed{s}' for s in range(5)],
 },
 'adamson':{
   '0.5':[f'gnn_v4sweep_adamson_pc0p5_seed{s}' for s in range(3)]+[f'gnn_v4_adamson_pc0p5_seed{s}' for s in (3,4)],
   '0.7':[f'gnn_v4sweep_adamson_pc0p7_seed{s}' for s in range(5)],
   '1':[f'gnn_v4sweep_adamson_pc1_seed{s}' for s in range(5)],
 },
}
paper={'rpe1':(0.651,0.402,0.216,0.280,0.042,2211),
       'k562':(0.766,0.570,0.322,0.248,0.022,2888),
       'adamson':(0.864,0.731,0.414,0.159,0.014,11)}
order=['ATE-rho','ATE-R2','Jaccard50','muWD_g0','FOR_g0','edges']
for ds,pcs in SETTINGS.items():
    p=paper[ds]
    print(f"\n{'='*82}\n{ds.upper()}  (paper ATE-rho={p[0]} R2={p[1]} Jac={p[2]} muWD={p[3]} FOR={p[4]} edges={p[5]})\n{'='*82}")
    for pc,cfgs in pcs.items():
        vals={m:[] for m in order}; n=0
        for cfg in cfgs:
            r=load(cfg)
            if r is None: print(f"    [missing {cfg}]"); continue
            n+=1
            for m in order: vals[m].append(r[m])
        line=f"  pc={pc:<5} n={n}  "
        for m in order:
            a=np.array(vals[m],float); mu=np.nanmean(a); sd=np.nanstd(a)
            if m=='edges': line+=f"{m}={mu:6.0f}±{sd:<4.0f} "
            else:          line+=f"{m}={mu:.3f}±{sd:.3f} "
        print(line)
