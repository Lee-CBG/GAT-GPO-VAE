# GAT-GPO-VAE

**Graph-Attention Perturbation Encoding for Explainable Modeling of Gene Perturbation Responses**

Seyedmasoud Mousavi, Jin G. Park, Heewook Lee — Arizona State University

[![Paper](https://img.shields.io/badge/paper-MLCB%202026-blue)](<PAPER_URL>)

---

## Overview

[GPO-VAE](https://github.com/dmis-lab/GPO-VAE) learns an interpretable gene×gene adjacency
matrix `Ŵ` as a gene regulatory network (GRN) alongside perturbation-response prediction —
but it only ever uses that GRN in the **loss**. Its Latent Perturbation Encoder maps each
gene's mask row through a shared row-wise MLP, so every gene is encoded independently and the
learned regulatory structure never informs the representation.

**GAT-GPO-VAE** replaces that MLP with a multi-layer graph attention network that propagates
over `σ(Ŵ)`, so a gene's perturbation embedding is formed from its regulatory neighborhood.
The edge weights entering the GAT are **detached** (stop-gradient), which keeps GRN learning
bit-for-bit identical to GPO-VAE while the GAT still exploits the graph in its forward pass.

Across three Perturb-seq datasets over five seeds:

| Dataset | Metric | GPO-VAE (reproduced) | GAT-GPO-VAE |
|---|---|---|---|
| RPE1 | ATE-ρ | 0.701 ± 0.018 | **0.725 ± 0.008** |
| | μWD | 0.283 ± 0.020 | **0.321 ± 0.013** |
| | FOR | 0.046 ± 0.006 | **0.038 ± 0.005** |
| K562 | ATE-ρ | 0.780 ± 0.009 | **0.797 ± 0.004** |
| | μWD | 0.245 ± 0.010 | **0.284 ± 0.013** |
| | FOR | 0.021 ± 0.006 | 0.021 ± 0.005 |
| Adamson | ATE-ρ | 0.860 ± 0.008 | 0.856 ± 0.006 (tie) |
| | μWD | 0.152 ± 0.011 | **0.219 ± 0.017** |
| | FOR | 0.010 ± 0.007 | 0.011 ± 0.004 |

μWD ↑ = mean Wasserstein distance over predicted edges; FOR ↓ = false omission rate. Both are
computed on the perturbed block `G°` only. Full baseline tables are in the paper.

---

## What changed relative to GPO-VAE

This repository is a fork of [dmis-lab/GPO-VAE](https://github.com/dmis-lab/GPO-VAE). The
model-side diff is small and deliberately localized:

| File | Change |
|---|---|
| `gpo_vae/models/gpo_vae/guides/gnn_guide.py` | **New.** `GATLayer`, `GNNEmbeddingEncoder`, `gpo_vae_GNNGuide`. Scratch-built GAT (no PyTorch Geometric dependency) replacing the row-wise MLP embedding encoder. |
| `gpo_vae/models/gpo_vae/__init__.py` | Registers `gpo_vae_GNNGuide`. |
| `gpo_vae/models/__init__.py` | Registers `gpo_vae_GNNGuide`. |
| `gpo_vae/models/utils/loss_modules.py` | Sparsity-penalty normalization fix (`probs.shape[0]**2`); KL loop skips non-distribution keys. The `T̂_K` GRN loss itself is **unchanged**. |
| `eval.py` | GRN metrics restricted to the `G°` block (no self-loops); `num_edges_full` retained for reference; wandb made optional. |

Everything downstream of the encoder — mask sampling, `Z_p = P(E ⊙ M)`, the artifact and basal
encoders, the decoder, and every loss term — is untouched.

### The detachment

The single most consequential line is in `gnn_guide.py`:

```python
edge_weight = torch.sigmoid(q_mask_logits).detach()   # v4 (default, paper)
# edge_weight = torch.sigmoid(q_mask_logits)          # v3 (no detach, ablation)
```

Without the `.detach()`, reconstruction gradients reach `Ŵ` through the attention mechanism and
GRN inference becomes seed-unstable — on RPE1 the edge count goes to 3687 ± 4084, a standard
deviation larger than the mean. Toggle these two lines to reproduce the ablation in the paper
(Appendix Table 3).

---

## Installation

```bash
git clone <REPO_URL>
cd <REPO_NAME>
conda env create -f environment.yml    # creates `gpo_vae_env`
conda activate gpo_vae_env
```

Tested on Python `<VERSION>`, PyTorch `<VERSION>`, Pyro `<VERSION>`, CUDA 13.0.

> If you already have a working GPO-VAE environment, this repo adds no new dependencies —
> the GAT is implemented in plain PyTorch.

---

## Data

We use the three Perturb-seq datasets from the GPO-VAE study:

| Dataset | Cells | Perturbed (`G°`) | Extended (`G⁺`) |
|---|---|---|---|
| Replogle K562 | 129,478 | 622 | 924 |
| Replogle RPE1 | 91,891 | 383 | 272 |
| Adamson | 46,236 | 68 | 347 |

Preprocessing follows [CausalBench](<CAUSALBENCH_URL>): weak perturbations removed
(DEGs > 50 by Anderson–Darling, knockdown efficiency ≤ −0.3, > 25 cells), cells filtered by
perturbation effect, perturbations with < 100 cells dropped. Quality-control labels follow
CRADLE-VAE's six criteria.

Download the preprocessed `.h5ad` files from `<DATA_URL>` and place them in `data/`:

```
data/
├── k562_qc_deg_matched_ctrl_idx_all_ot.h5ad
├── <rpe1_filename>.h5ad
└── <adamson_filename>.h5ad
```

**Split:** cell-level random 80/16/4 train/val/test, following GPO-VAE and CRADLE-VAE. All
perturbations appear in all splits — this benchmark measures per-perturbation average
treatment effect estimation on held-out **cells**, not zero-shot generalization to unseen
perturbations.

---

## Quickstart

Training entry points are per-dataset:

```bash
# RPE1
CUDA_VISIBLE_DEVICES=0 python train_rpe1.py --config ./demo/gnn_v4_rpe1_pc15_seed0.yaml

# K562
CUDA_VISIBLE_DEVICES=0 python train_replogle.py --config ./demo/gnn_v4_k562_pc7_seed0.yaml

# Adamson
CUDA_VISIBLE_DEVICES=0 python train_adamson.py --config ./demo/gnn_v4_adamson_pc0p5_seed0.yaml
```

Evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled python eval.py \
    --experiment_path results/gnn_v4_rpe1_pc15_seed0 \
    --perturbseq \
    --batch_size 128 \
    --ate_n_particles 2500 \
    --devices 0 \
    --thr 3
```

Aggregate across seeds:

```bash
python aggregate_results.py --pattern 'gnn_v4_rpe1_pc15_seed*'
```

---

## Reproducing the paper

| Paper element | Script / configs |
|---|---|
| Table 1, Appendix Table 1 (main comparison) | `demo/gnn_v4_<ds>_pc<X>_seed{0-4}.yaml` + `demo/gpo_vae_<ds>_seed{0-4}.yaml`; `run_all_evals.sh`; `aggregate_results.py` |
| Appendix Table 3 (detachment ablation) | Toggle the `.detach()` line in `gnn_guide.py`, retrain at matched `penaly_coeff` |
| Appendix Table 4 (architecture ablation) | `demo/gnn_arch_<ds>_<L1\|L3\|H2\|H8\|D128\|D512\|drop0p1\|drop0p3>_seed{0-2}.yaml` |
| Sparsity-coefficient selection (Sec. 4.4) | `demo/gnn_v4sweep_<ds>_pc<X>_seed<N>.yaml`; `run_v4sweep.sh`, `run_v4sweep_eval.sh`, `aggregate_v4sweep.py`, `aggregate_5seed.py` |
| Appendix Table 5 (directed-bridge collapse) | `directed_bridge_sweep.py` |
| Table 2, Figure 3 (biological modules) | See [Biological analysis](#biological-analysis) below |

---

## Configuration reference

Shared hyperparameters (retained from GPO-VAE, unchanged):

```yaml
lr: 0.0003
batch_size: 512
n_particles: 5
n_latent: 200
mask_prior: 0.3
mask_init: 0
gloss_coeff: 100        # λ_g, K-hop DGE loss
hop: 5                  # K
fc_criteria: 0.5
beta: 0.1               # artifact disentanglement
max_epochs: 2000
early_stop_patience: 30
```

GAT-specific (defaults; the architecture ablation shows results are insensitive to these):

```yaml
gnn_n_layers: 2
gnn_n_heads: 4
gnn_d_hidden: 256
gnn_dropout: 0.0
```

Sparsity coefficient — the **only** per-dataset hyperparameter we select:

| Dataset | GAT-GPO-VAE | GPO-VAE baseline |
|---|---|---|
| RPE1 | `penaly_coeff: 15` | `penaly_coeff: 10` |
| K562 | `penaly_coeff: 7` | `penaly_coeff: <VALUE>` |
| Adamson | `penaly_coeff: 0.5` | `penaly_coeff: 1` |

> The YAML key is spelled `penaly_coeff` (sic) — inherited from the upstream codebase.
> Preserved for config compatibility.

---

## Biological analysis

The convergence-subnetwork pipeline behind Section 4.7 and Table 2. Run in order:

```bash
# 1. Recover per-edge Wasserstein distances with (parent, child) identity preserved
python extract_edge_wd.py --experiment_path results/gnn_v4_k562_pc7_seed4

# 2. Build convergence subnetworks: for each extended gene, the perturbed genes
#    converging on it (prob > 0.6, WD >= 0.3, 3-30 parents)
python subnetworks_v2.py --min_parents 3 --max_parents 30 --prob_thr 0.6 --wd_thr 0.3

# 3. Over-representation analysis (KEGG_2021_Human + Reactome_2022 via Speedrichr),
#    using the 622 perturbed genes as custom background
python enrich_subnets_bg.py

# 4. Per-subnetwork biologist report; flags nonspecific pathway terms
python build_handoff.py

# 5. Cross-seed module reproducibility, matched by gene-set content
python aggregate_subnets.py
```

**Enrichment protocol.** Over-representation analysis against the `G°` perturbed genes as
background (not the whole genome) — a genome background inflates significance through
essential-gene composition bias. Benjamini–Hochberg FDR < 0.05. GO libraries are excluded:
the API returns 500s on unrecognized non-coding gene symbols. Coverage is tracked per
subnetwork so API failures are not silently counted as non-significant.

**A note on module grouping.** Subnetworks are grouped into biological *programs* by shared
gene-set content, not by anchor identity — anchors are frequently non-coding and unstable
across seeds. Subnetworks with no significant non-flagged term are reported but not annotated.

---

## Reproducibility notes

These are the traps that cost us the most time. Please read before reporting a mismatch.

**1. The evaluation harness has two load-bearing lines.** In
`gpo_vae/models/utils/predictor.py`:

```python
# line ~465 — MUST be test-split only. Using all cells inflates ATE.
adata = data_module.adata[data_module.adata.obs["split"] == "test"]

# line ~469 — MUST be 'FC3'. Using 'highly_variable' runs the network
# statistical evaluation over the wrong gene universe and inflates K562 FOR ~3x.
criteria = 'FC3'
```

Both must match the upstream original. Under this harness our GPO-VAE reproduction recovers
the published baseline on μWD, FOR and edge count across all three datasets.

**2. GRN metrics are `G°`-restricted.** μWD, FOR and edge counts are computed on the
perturbed block only (`adj_matrix[:pert_gene_len, :pert_gene_len]`, self-loops removed), for
comparability with causal-discovery baselines. The full matrix count is reported separately as
`num_edges_full`. The `G⁺` block is saturated (~800k edges > 0.5) in **both** models and is not
a meaningful quality signal.

**3. μWD is `positive_mean_wasserstein`.** `output_graph['wasserstein_distance']['mean']`.
`negative_mean_wasserstein` is a different quantity.

**4. Lightning directory auto-increment.** Re-running a config whose `name` already exists
creates `<name>-2`, `<name>-3`, etc. rather than overwriting. Aggregation scripts select the
highest-epoch checkpoint; verify you are evaluating the run you think you are.

**5. Run eval with `WANDB_MODE=disabled`** unless you have configured wandb.

---

## Hardware and runtime

Developed on 4× NVIDIA L40S (46 GB each), CUDA 13.0.

| | Training (peak) | Evaluation (peak) |
|---|---|---|
| GPO-VAE | ~0.5 GB | — |
| GAT-GPO-VAE, K562 | ~19.6 GB | ~43 GB |
| GAT-GPO-VAE, RPE1 | `<VALUE>` GB | `<VALUE>` GB |
| GAT-GPO-VAE, Adamson | `<VALUE>` GB | ~13 GB |

The GAT is substantially more memory-hungry than the MLP it replaces. Two implementation
choices dominate: the attention map is a dense `[B, n, n, H]` tensor computed over all `n²`
gene pairs regardless of edge sparsity, and the five particles are processed sequentially
rather than batched. Both are reducible — the learned GRN is sparse by construction, so a
sparse-edge implementation would cut the dominant term.

Practical limits: one K562 evaluation per GPU. K562 with 8 attention heads requires
`--batch_size 64` (verified to leave metrics unchanged: μWD and edge counts are bit-identical
to `--batch_size 128`, ATE differs only in the fourth decimal).

---

## Repository layout

```
├── gpo_vae/
│   └── models/
│       ├── gpo_vae/guides/gnn_guide.py     # GAT encoder (new)
│       └── utils/
│           ├── loss_modules.py             # sparsity-penalty + KL fixes
│           └── predictor.py                # evaluation harness
├── demo/                                   # all training configs
├── train_rpe1.py / train_replogle.py / train_adamson.py
├── eval.py
├── extract_edge_wd.py                      # per-edge Wasserstein distances
├── subnetworks_v2.py                       # convergence subnetworks
├── enrich_subnets_bg.py                    # KEGG/Reactome ORA
├── build_handoff.py                        # per-subnetwork report
├── aggregate_subnets.py                    # cross-seed module recovery
├── directed_bridge_sweep.py                # directed-bridge collapse analysis
├── aggregate_results.py / aggregate_5seed.py / aggregate_v4sweep.py
└── run_all_evals.sh / run_v4sweep.sh / run_v4sweep_eval.sh
```

---

## Citation

```bibtex
@inproceedings{mousavi2026gatgpovae,
  title     = {{GAT-GPO-VAE}: Graph-Attention Perturbation Encoding for
               Explainable Modeling of Gene Perturbation Responses},
  author    = {Mousavi, Seyedmasoud and Park, Jin G. and Lee, Heewook},
  booktitle = {Machine Learning in Computational Biology (MLCB)},
  year      = {2026}
}
```

Please also cite the base method:

```bibtex
@article{gpovae2025,
  title   = {{GPO-VAE}: Modeling Explainable Gene Perturbation Responses
             utilizing GRN-Aligned Parameter Optimization},
  author  = {<AUTHORS>},
  journal = {arXiv preprint arXiv:2501.18973},
  year    = {2025}
}
```

---

## License

`<LICENSE>` — note that this repository is a derivative of
[dmis-lab/GPO-VAE](https://github.com/dmis-lab/GPO-VAE); please check the upstream license
terms before redistributing.

## Acknowledgments

We thank the authors of GPO-VAE for releasing their code and datasets. This work builds
directly on their implementation.
