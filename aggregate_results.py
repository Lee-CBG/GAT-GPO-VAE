import pandas as pd, numpy as np

configs = {
    "GPO-VAE RPE1":   [f"results/gpo_vae_rpe1_seed{s}-3" for s in range(5)],
    "GNN v4 RPE1":    ["results/gnn_vae_rpe1_pc15_seed0-2"] + [f"results/gnn_v4_rpe1_pc15_seed{s}" for s in range(1,5)],
    "GPO-VAE K562":   [f"results/gpo_vae_replogle_seed{s}" for s in range(5)],
    "GNN v4 K562":    ["results/gnn_vae_k562_pc5_seed0-3"] + [f"results/gnn_v4_k562_pc5_seed{s}" for s in range(1,5)],
    "GPO-VAE Adamson":[f"results/gpo_vae_adamson_pc1_seed{s}" for s in range(5)],
    "GNN v4 Adamson": [f"results/gnn_v4_adamson_pc0p5_seed{s}" for s in range(5)],
}
metrics = ["ATE_pearsonr-test","ATE_r2-test","jaccard_sim_top50-test",
           "positive_mean_wasserstein__g0","false_omission_rate__g0","num_edges"]

for name, dirs in configs.items():
    rows = []
    for d in dirs:
        try:
            df = pd.read_csv(f"{d}/test_metrics.csv", index_col=0)
            rows.append({m: float(df.loc[m, "0"]) for m in metrics if m in df.index})
        except Exception as e:
            print(f"  MISSING/ERR {d}: {e}")
    if not rows:
        print(f"{name}: no results\n"); continue
    agg = pd.DataFrame(rows)
    print(f"=== {name} (n={len(rows)}) ===")
    for m in metrics:
        if m in agg.columns:
            print(f"  {m:<32} {agg[m].mean():.4f} +/- {agg[m].std():.4f}")
    print()
