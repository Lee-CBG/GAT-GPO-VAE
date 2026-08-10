import pandas as pd

# explicit dirs so the retrained K562 seeds point at the -3 dirs
configs = {
    "GNN v3 RPE1": [f"results/gnn_v3_rpe1_pc15_seed{s}" for s in range(5)],
    "GNN v3 K562": ["results/gnn_v3_k562_pc5_seed0-3",
                    "results/gnn_v3_k562_pc5_seed1",
                    "results/gnn_v3_k562_pc5_seed2-3",
                    "results/gnn_v3_k562_pc5_seed3-3",
                    "results/gnn_v3_k562_pc5_seed4"],
    "GNN v3 Adamson": [f"results/gnn_v3_adamson_pc0p5_seed{s}" for s in range(5)],
}
metrics = ["ATE_pearsonr-test","ATE_r2-test","jaccard_sim_top50-test",
           "positive_mean_wasserstein__g0","false_omission_rate__g0","num_edges"]

for name, dirs in configs.items():
    rows = []
    for d in dirs:
        try:
            df = pd.read_csv(f"{d}/test_metrics.csv", index_col=0)
            rows.append({m: float(df.loc[m,"0"]) for m in metrics if m in df.index})
        except Exception as e:
            print(f"  {d}: {e}")
    if not rows:
        print(f"{name}: no results\n"); continue
    agg = pd.DataFrame(rows)
    print(f"=== {name} (n={len(rows)}) ===")
    for m in metrics:
        if m in agg.columns:
            print(f"  {m:<32} {agg[m].mean():.4f} +/- {agg[m].std():.4f}")
    print()
