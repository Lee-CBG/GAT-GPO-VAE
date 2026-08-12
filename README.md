# GAT-GPO-VAE

**Graph-Attention Perturbation Encoding for Explainable Modeling of Gene Perturbation Responses**

Seyedmasoud Mousavi, Jin G. Park, Heewook Lee — Arizona State University

> **Status:** manuscript under review at MLCB 2026. A citation will be added here upon acceptance.

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

This repository is a fork of [dmis-lab/GPO-VAE](https://github.com/dmis-lab/GPO-VAE), with the
upstream commit history preserved. `git diff 4e7c3ad HEAD` shows the complete change set. The
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
edge_weight = torch.sigmoid(q_mask_logits).detach()   # detached (default, paper)
# edge_weight = torch.sigmoid(q_mask_logits)          # no detach (ablation)
```

Without the `.detach()`, reconstruction gradients reach `Ŵ` through the attention mechanism and
GRN inference becomes seed-unstable — on RPE1 the edge count goes to 3687 ± 4084, a standard
deviation larger than the mean. Toggle these two lines to reproduce the ablation in the paper.

---

## Installation

```bash
git clone https://github.com/Lee-CBG/GAT-GPO-VAE.git
cd GAT-GPO-VAE
conda env create -f environment.yml
conda activate gpo_vae_env
```

Key versions: Python 3.9.15, PyTorch 2.4.0, Pyro 1.9.1, numpy 1.26.4, scanpy 1.10.2, CUDA 13.0.

> This fork adds no new dependencies — the GAT is implemented in plain PyTorch. An existing
> GPO-VAE environment will work as-is.

---

## Data

We use the three Perturb-seq datasets from the GPO-VAE study:

| Dataset | Cells | Perturbed (`G°`) | Extended (`G⁺`) |
|---|---|---|---|
| Replogle K562 | 129,478 | 622 | 924 |
| Replogle RPE1 | 91,891 | 383 | 272 |
| Adamson | 46,236 | 68 | 347 |

Download the preprocessed files using the links published by the GPO-VAE authors
(~11 GB total, so they are not included in this repository):

```bash
pip install gdown
gdown https://drive.google.com/uc?id=1gpnjtKYLAsyGrPqGT8NbTcvx2tiLowXN   # datasets
tar -zxvf datasets.tar.gz
```

This produces:

```
datasets/
├── k562_qc_deg_matched_ctrl_idx_all_ot.h5ad
├── rpe1_qc_deg_matched_ctrl_idx_all_ot.h5ad
└── adamson_qc_deg_matched_ctrl_idx_all_ot.h5ad
```

The Replogle summary statistics (`summary_stats.xlsx`, `summary_stats_adamson.csv`) are already
included in this repository under `summary_stats/`, so the second download from the upstream
instructions is not needed.

> **Path gotcha.** The shipped configs set
> `data_module_kwargs.stat_path: ./datasets/summary_stats.xlsx`, but the file lives in
> `summary_stats/`. Either copy it into `datasets/` or edit `stat_path` in your config before
> training. If `stat_path` cannot be read, the CausalBench filtering step is where the run will
> fail — and if it is silently skipped, gene counts will not match those above.

**Preprocessing.** Follows [CausalBench](https://github.com/causalbench/causalbench): weak
perturbations removed (DEGs > 50 by Anderson–Darling, knockdown efficiency ≤ −0.3, > 25 cells),
cells filtered by perturbation effect, perturbations with < 100 cells dropped. Quality-control
labels follow CRADLE-VAE's six criteria.

**Split.** Cell-level 64/16/20 train/val/test, following GPO-VAE and CRADLE-VAE. It is produced
by two nested calls in `gpo_vae/data/<dataset>/data_module.py` (identical in all three datasets):

```python
train_idx, test_idx = train_test_split(idx, train_size=0.8, random_state=0)       # 80 / 20
train_idx, val_idx  = train_test_split(train_idx, train_size=0.8, random_state=0) # 64 / 16
```

The second call re-splits **train**, not test, so train = 0.8 x 0.8 = 64%, val = 0.8 x 0.2 = 16%,
test = 20%. `random_state=0` is fixed and the assignment is written to `adata.obs["split"]` in the
preprocessed `.h5ad`, so **the split is identical across every seed, model and run in this
repository**; seeds vary model initialization only.

| dataset | train | val | test |
|---|---|---|---|
| RPE1 | 58,809 | 14,703 | 18,379 |
| K562 | 82,865 | 20,717 | 25,896 |
| Adamson | 29,590 | 7,398 | 9,248 |

All perturbations appear in all splits (RPE1: 384/384 with cells in each) — this benchmark
measures per-perturbation average treatment effect estimation on held-out **cells**, not
zero-shot generalization to unseen perturbations.

> *Correction:* an earlier version of this README gave the split as 80/16/4. That was a
> documentation error — the second `train_test_split` was misread as partitioning the test set
> rather than the training set. No data, split, or result changed.

---

## Quickstart

Training entry points are per-dataset:

```bash
# RPE1
CUDA_VISIBLE_DEVICES=0 python train_rpe1.py --config ./demo/gnn_v4sweep_rpe1_pc15_seed0.yaml

# K562
CUDA_VISIBLE_DEVICES=0 python train_replogle.py --config ./demo/gnn_v4sweep_k562_pc7_seed0.yaml

# Adamson
CUDA_VISIBLE_DEVICES=0 python train_adamson.py --config ./demo/gnn_v4_adamson_pc0p5_seed0.yaml
```

Evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled python eval.py \
    --experiment_path results/gnn_v4sweep_rpe1_pc15_seed0 \
    --perturbseq \
    --batch_size 128 \
    --ate_n_particles 2500 \
    --devices 0 \
    --thr 3
```

Aggregate across seeds with `aggregate_results.py`.

---

## Reproducing the paper

Final operating points are RPE1 `penaly_coeff=15`, K562 `penaly_coeff=7`, Adamson
`penaly_coeff=0.5`.

| Paper element | Configs / scripts |
|---|---|
| Main comparison | `demo/gnn_v4sweep_rpe1_pc15_seed{0,1,2}.yaml` + `demo/gnn_v4_rpe1_pc15_seed{3,4}.yaml`, `demo/gnn_v4sweep_k562_pc7_seed{0-4}.yaml`, `demo/gnn_v4_adamson_pc0p5_seed{0-4}.yaml`; baselines `demo/gpo_vae_{rpe1,replogle,adamson_pc1}_seed{0-4}.yaml`; `run_all_evals.sh`, `aggregate_results.py` |
| Detachment ablation | `demo/gnn_v3_*_seed{0-4}.yaml` (no-detach variant), `aggregate_v3.py` |
| Architecture ablation | `demo/gnn_arch_<ds>_<L1\|L3\|H2\|H8\|D128\|D512\|drop0p1\|drop0p3>_seed{0-2}.yaml`; `run_arch_sweep.sh`, `run_arch_eval.sh`, `aggregate_arch.py` |
| Sparsity-coefficient selection | `demo/gnn_v4sweep_<ds>_pc<X>_seed<N>.yaml`; `run_v4sweep.sh`, `run_v4sweep_eval.sh`, `aggregate_v4sweep.py`, `aggregate_5seed.py` |
| Directed-bridge collapse | `directed_bridge_sweep.py` |
| Biological modules | See [Biological analysis](#biological-analysis) |

> Note on config naming: `gnn_v4*` = the detached model reported in the paper; `gnn_v3*` = the
> no-detach ablation; `gnn_vae_*` are earlier runs retained for provenance and are **not** the
> paper's configuration.
> The five reported RPE1 seeds come from two run families: `gnn_v4sweep_*` supplies seeds 0–2
> and `gnn_v4_*` supplies seeds 3–4, as neither family contains all five. Both use identical
> configurations and the same detached encoder.

---

## STRING PPI validation

`string_inputs/` contains the gene sets used for the protein–protein interaction
check reported in Section 4.4, Table 3, and Appendix Figure 5. One file per
convergence subnetwork.

STRING was queried through the web interface rather than scripted, so these files
are the exact inputs and the results can be re-derived without running our code:

1. https://string-db.org → Search → Multiple proteins
2. Paste the contents of one `subnetNN_ANCHOR.txt`
3. Organism: Homo sapiens
4. Settings: default (minimum interaction score 0.400, all active sources)
5. Read the PPI enrichment p-value from the Analysis tab

`subnet_summary.csv` gives, for all 16 subnetworks: anchor, parent count, median
edge Wasserstein distance, candidate pathway annotation, STRING PPI enrichment
p-value, average node degree, and our verdict.

**Scope notes.** Anchor genes are extended-gene (G+) nodes and are frequently
non-coding; the PPI test asks whether the converging perturbed genes are coherent
with each other. STRING uses its default whole-genome background — the custom
background of the 622 perturbed genes applies to the KEGG/Reactome pathway
enrichment only (`enrich_subnets_bg.py`), where the panel's enrichment for
essential genes would otherwise inflate significance.

Gene symbols follow the dataset's original annotation; STRING resolves legacy symbols automatically (e.g. RARS → RARS1).

Thirteen of sixteen subnetworks reach p < 0.01. The three that do not (CCNJ,
ZNF32, PHF10) are reported in the paper and flagged as likely artifacts.

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

GAT-specific defaults (the architecture ablation shows results are insensitive to these):

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
| K562 | `penaly_coeff: 7` | `penaly_coeff: 5` |
| Adamson | `penaly_coeff: 0.5` | `penaly_coeff: 1` |

> The YAML key is spelled `penaly_coeff` (sic), inherited from the upstream codebase and
> preserved for config compatibility.

---

## Biological analysis

The convergence-subnetwork pipeline behind the paper's biological module analysis. Run in order:

```bash
# 1. Per-edge Wasserstein distances with (parent, child) identity preserved
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled python extract_edge_wd.py \
    --experiment_path results/gnn_v4sweep_k562_pc7_seed4

# 2. Convergence subnetworks: for each extended gene, the perturbed genes
#    converging on it (prob > 0.6, WD >= 0.3, 3-30 parents)
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled python subnetworks_v2.py \
    --experiment_path results/gnn_v4sweep_k562_pc7_seed4

# 3. Over-representation analysis (KEGG_2021_Human + Reactome_2022 via Speedrichr)
python enrich_subnets_bg.py

# 4. Per-subnetwork report with nonspecific-term flagging
python build_handoff.py

# 5. Cross-seed module reproducibility, matched by gene-set content
python aggregate_subnets.py
```

**Enrichment protocol.** Over-representation analysis against the `G°` perturbed genes as
background (not the whole genome) — a genome background inflates significance through
essential-gene composition bias. Benjamini–Hochberg FDR < 0.05. GO libraries are excluded: the
API returns 500s on unrecognized non-coding gene symbols. Coverage is tracked per subnetwork so
API failures are not silently counted as non-significant.

**Module grouping.** Subnetworks are grouped into biological *programs* by shared gene-set
content, not by anchor identity — anchors are frequently non-coding and unstable across seeds.
Subnetworks with no significant non-flagged term are reported but not annotated.

Related scripts: `subnetworks.py` and `subnetworks_g0.py` implement earlier and `G°`-only
variants of the subnetwork definition; `threshold_sweep.py` varies the convergence thresholds.

---

## Reproducibility notes

These are the traps that cost us the most time. Please read before reporting a mismatch.

**1. The evaluation harness has two load-bearing lines.** In
`gpo_vae/models/utils/predictor.py`:

```python
# MUST be test-split only. This governs the NETWORK statistical evaluation
# (per-edge Wasserstein distance and FOR), NOT ATE — see note 2 below.
# Using all cells here inflates the GRN metrics.
adata = data_module.adata[data_module.adata.obs["split"] == "test"]

# MUST be 'FC3'. Using 'highly_variable' runs the network statistical
# evaluation over the wrong gene universe and inflates K562 FOR ~3x.
criteria = 'FC3'
```

Both match the upstream original. Under this harness our GPO-VAE reproduction recovers the
published baseline on μWD, FOR and edge count across all three datasets.

**2. The ATE reference is computed from all cells, and this is not fixed here.**
`eval.py` (~L66) calls `get_estimated_average_treatment_effects()` with no `split` argument, so
the reference average treatment effect every model is scored against is built from all cells —
including the 64% used for training. The `split` parameter exists on that method but is never
passed. Consequently the `ATE_pearsonr-all`, `-train`, `-val` and `-test` keys in
`test_metrics.csv` are **bit-identical**: the split applied at `eval.py` (~L89) is by
*perturbation*, and every perturbation appears in every split, so all four are the same number
under four names.

This is inherited from the upstream GPO-VAE benchmark and affects every published number on it
equally, ours and the baseline's. **We have deliberately not changed it**, because the code as-is
is what reproduces the published values. To score against a leakage-free reference, use
`rescore_ate.py`, which reports both.

**3. GRN metrics are `G°`-restricted.** μWD, FOR and edge counts are computed on the perturbed
block only (`adj_matrix[:pert_gene_len, :pert_gene_len]`, self-loops removed), for comparability
with causal-discovery baselines. The full-matrix count is reported separately as
`num_edges_full`. The `G⁺` block is saturated (~800k edges > 0.5) in **both** models and is not
a meaningful quality signal.

**4. μWD is `positive_mean_wasserstein`.** `output_graph['wasserstein_distance']['mean']`.
`negative_mean_wasserstein` is a different quantity.

**5. Lightning directory auto-increment.** Re-running a config whose `name` already exists
creates `<name>-2`, `<name>-3`, etc. rather than overwriting. Verify you are evaluating the run
you think you are; `grn.csv` should have 1547 columns for K562 (1546 genes + index).

**6. Run eval with `WANDB_MODE=disabled`** unless you have configured wandb.

---

## Leakage-free re-scoring and non-parametric baselines

These three scripts support the evaluation analysis in our author response. They require no
retraining — the model ATE is a deterministic function of the checkpoint.

```bash
python rescore_ate.py <run_dir> [n_particles]   # default 2500
python linear_baselines.py <rpe1|k562|adamson>
python reliability_check.py <rpe1|k562|adamson>
```

- **`rescore_ate.py`** regenerates the model ATE from a checkpoint and scores it against *both*
  the all-cells reference (which reproduces the value in that run's `test_metrics.csv`, and so
  doubles as a correctness check) and a test-cells-only reference. Always confirm the all-cells
  block matches before trusting the test-only block.
- **`linear_baselines.py`** scores two non-parametric predictors against the test-cells-only
  reference: a per-perturbation mean over training cells, and a global training mean. Writes
  `linear_baselines_<dataset>.json`.
- **`reliability_check.py`** estimates the measurement ceiling by splitting the training cells in
  half and scoring one half's estimated effects against the other, plus test-vs-val and
  test-vs-train comparisons.

All three use the `perturbseq` scale (normalize to 1e4 UMI, `log2(x+1)`, *then* difference
against the control mean) — the same scale `eval.py --perturbseq` uses. The `mean` scale
differences raw counts instead and gives substantially different values.

**Resource note.** These materialize the full expression matrix in float64 and need roughly
33 GB of host RAM per process (plus ~1.4 GB GPU for `rescore_ate.py`). Run at most two or three
K562 or Adamson re-scores concurrently.

---

## Hardware and runtime

Developed on 4× NVIDIA L40S (46 GB each), CUDA 13.0.

Peak **allocated** GPU memory (`torch.cuda.max_memory_allocated`), measured with
`profile_memory.py` and `profile_eval.py`. Identical settings for both models: batch size 512
with 5 particles in training, batch size 128 with 2500 particles at evaluation.

| dataset | model | params | Training | Evaluation |
|---|---|---|---|---|
| RPE1 (655 genes) | GPO-VAE | 8.78 M | 3.13 GB | 6.10 GB |
| RPE1 | GAT-GPO-VAE | 8.06 M | 3.46 GB | 6.59 GB |
| K562 (1546 genes) | GPO-VAE | 12.33 M | 4.87 GB | 16.23 GB |
| K562 | GAT-GPO-VAE | 11.10 M | 6.70 GB | 26.29 GB |

The GAT costs 1.1–1.6x the memory of the MLP it replaces, and the overhead grows with gene
count (1.08x at 655 genes, 1.62x at 1546) because the attention map is a dense `[B, n, n, H]`
tensor computed over all `n²` gene pairs regardless of edge sparsity, and the five particles are
processed sequentially rather than batched. Both are reducible — the learned GRN is sparse by
construction, so a sparse-edge implementation would cut that term.

Note that both models are dominated by the `n x n` GRN parameter matrix and the shared decoder,
not by the encoder: the MLP baseline alone needs 16 GB to evaluate 1546 genes. The GAT also has
*fewer* parameters than the MLP it replaces, and is slightly faster at evaluation (600 s vs
666 s on K562, 134 s vs 158 s on RPE1).

**Measurement convention.** These are *allocated* figures. Peak *reserved* memory (what
`nvidia-smi` shows, including the CUDA context and the caching allocator's unreturned blocks)
runs 1.3–1.6x higher — 41.5 GB reserved against 26.3 GB allocated for K562 GAT evaluation. Size
your GPUs against the reserved figure.

Practical limits: one K562 evaluation per GPU. K562 with 8 attention heads requires
`--batch_size 64` (verified to leave metrics unchanged: μWD and edge counts are bit-identical to
`--batch_size 128`, ATE differs only in the fourth decimal).

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
├── summary_stats/                          # Replogle summary statistics
├── train_rpe1.py / train_replogle.py / train_adamson.py
├── eval.py
├── extract_edge_wd.py                      # per-edge Wasserstein distances
├── subnetworks_v2.py                       # convergence subnetworks
├── enrich_subnets_bg.py                    # KEGG/Reactome ORA
├── build_handoff.py                        # per-subnetwork report
├── aggregate_subnets.py                    # cross-seed module recovery
├── directed_bridge_sweep.py                # directed-bridge collapse analysis
├── threshold_sweep.py                      # convergence-threshold sensitivity
├── aggregate_*.py                          # results aggregation
└── run_*.sh                                # multi-GPU job schedulers
```

---

## Citation

This manuscript is under review. A citation will be added here upon acceptance.

Please cite the base method:

```bibtex
@article{gpovae2025,
  title   = {{GPO-VAE}: Modeling Explainable Gene Perturbation Responses
             utilizing GRN-Aligned Parameter Optimization},
  author  = {Baek, Seungheun and Park, Soyon and Chok, Yan Ting and
             Gim, Mogan and Kang, Jaewoo},
  journal = {arXiv preprint arXiv:2501.18973},
  year    = {2025}
}
```

---

## License

This repository is a derivative work of
[dmis-lab/GPO-VAE](https://github.com/dmis-lab/GPO-VAE), released under the
**Creative Commons Attribution-NonCommercial 4.0 International** license
(© 2026 Seungheun Baek, Soyon Park, Yan Ting Chok, Mogan Gim and Jaewoo Kang). The original
`LICENSE` is retained unmodified, and the same terms apply to this fork. Modifications made in
this repository are described in [What changed relative to GPO-VAE](#what-changed-relative-to-gpo-vae).

## Acknowledgments

We thank the authors of GPO-VAE for releasing their code and datasets. This work builds directly
on their implementation.
