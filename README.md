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

Across three Perturb-seq datasets over five seeds, under the evaluation protocol inherited from
GPO-VAE:

| Dataset | Metric | GPO-VAE (reproduced) | GAT-GPO-VAE |
|---|---|---|---|
| RPE1 | ATE-ρ | 0.701 ± 0.018 | **0.725 ± 0.008** |
| | μWD | 0.283 ± 0.020 | 0.321 ± 0.013 |
| | FOR | 0.046 ± 0.006 | 0.038 ± 0.005 |
| K562 | ATE-ρ | 0.780 ± 0.009 | **0.797 ± 0.004** |
| | μWD | 0.245 ± 0.010 | 0.284 ± 0.013 |
| | FOR | 0.021 ± 0.006 | 0.021 ± 0.005 |
| Adamson | ATE-ρ | 0.860 ± 0.008 | 0.856 ± 0.006 (tie) |
| | μWD | 0.152 ± 0.011 | 0.219 ± 0.017 |
| | FOR | 0.010 ± 0.007 | 0.011 ± 0.004 |

ATE-ρ ↑ = Pearson correlation between predicted and measured average treatment effects.
μWD ↑ = mean Wasserstein distance over predicted edges; FOR ↓ = false omission rate. Both GRN
metrics are computed on the perturbed block `G°` only. Full baseline tables are in the paper.

> **The μWD column is not evidence of GRN quality, and we no longer claim it is.** μWD evaluated
> at each model's own operating point is close to a monotone decreasing function of edge count, so
> a method can improve it by inferring fewer edges. At matched edge count the advantage
> disappears. See [Evaluation findings](#evaluation-findings) — this correction, and two others,
> came out of peer review and are described there in full.

---

## Evaluation findings

Three properties of this benchmark family surfaced while preparing our author response. All three
concern the evaluation, not the architecture, and none was introduced by this fork. They are
recorded here because the numbers above cannot be read correctly without them.

**1. μWD tracks edge count, not architecture.** Ranking all candidate edges by predicted
probability and averaging WD over the top K, with the same K for both models:

| K | RPE1 GAT | RPE1 GPO-VAE | K562 GAT | K562 GPO-VAE |
|---|---|---|---|---|
| 500 | 0.501 | 0.501 | 0.395 | **0.425** |
| 1000 | 0.380 | 0.378 | 0.320 | **0.335** |
| 1500 | 0.323 | 0.323 | 0.285 | **0.296** |
| 2000 | 0.292 | 0.290 | 0.263 | **0.271** |
| 2500 | 0.271 | 0.268 | 0.249 | **0.255** |
| 3000 | 0.254 | 0.251 | 0.238 | **0.242** |

Seed standard deviations are 0.002–0.011. RPE1 is indistinguishable at every K; K562 favors the
baseline at every K. Extending this to a frontier over 43 trained runs spanning 536–4,279 edges,
the mean difference against the baseline's curve at matched edge count is −0.0001 on RPE1 (10 of
21 runs above) and −0.0092 on K562 (4 of 17). Our no-detachment ablation — including the seed that
diverges to 10,971 edges — lies on the same curve. Three architectures, one curve. Our operating
points use larger sparsity penalties than the baseline's (15 vs 10 on RPE1, 7 vs 5 on K562), which
is what produces the apparent gap. Both models rank far above chance, so the signal is real; it is
just not architecture-dependent. Reproduce with `matched_k_mu_wd.py` and `mu_wd_frontier.py`.

**2. The ATE reference is not held out.** See [Reproducibility note 2](#reproducibility-notes)
for the mechanism and its provenance. Re-scored against a test-cells-only reference over five
seeds with an exact paired permutation test, the architectural comparison is unchanged in
direction and significance:

| dataset | GAT-GPO-VAE | GPO-VAE | seeds won | p |
|---|---|---|---|---|
| RPE1 | **0.558 ± 0.006** | 0.540 ± 0.013 | 5/5 | 0.031 |
| K562 | **0.609 ± 0.004** | 0.596 ± 0.006 | 5/5 | 0.031 |
| Adamson | 0.743 ± 0.002 | 0.747 ± 0.007 | 2/5 | 0.84 |

`p = 0.031` is the smallest value attainable at n = 5. The held-out reference deflates both models
by 0.11–0.19 but preserves the ordering and the Adamson tie.

**3. A non-parametric baseline saturates the metric.** Against the same held-out reference:

| dataset | per-perturbation train mean | reliability ceiling | test vs val | global train mean | GAT-GPO-VAE |
|---|---|---|---|---|---|
| RPE1 | 0.658 | 0.661 | 0.515 | 0.351 | 0.558 |
| K562 | 0.644 | 0.647 | 0.494 | 0.319 | 0.609 |
| Adamson | 0.814 | 0.818 | 0.707 | 0.476 | 0.743 |

The reliability ceiling is one half of the training cells scored against the other half — two
independent measurements of the same effects, no model involved. The per-perturbation mean lands
on it. It is not modeling anything; it is direct measurement, and on this benchmark direct
measurement is not beaten by either deep model. The global train mean is far below everything, so
this reflects perturbation-specific signal rather than a metric artifact. Reproduce with
`linear_baselines.py` and `reliability_check.py`.

None of this affects the controlled GAT-vs-MLP comparison, which is verified under both
references. It bounds what that comparison means.

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

The stop-gradient blocks only the gradient flowing back into `Ŵ` along the attention path.
Message passing is untouched: every embedding is still formed from the gene's regulatory
neighborhood, and the GAT's own projections and attention vectors train normally on
reconstruction. What detachment removes is the model's ability to reshape the causal graph to
suit its own message passing.

**The resulting instability is dataset-specific, and we state it narrowly.** On RPE1, no-detach
gives an edge count of 3687 ± 4084 over five seeds — but that spread is driven by a single seed
that diverges to 10,971 edges (ATE-ρ 0.378). Excluding it, no-detach reaches 0.711 ± 0.018
against 0.725 ± 0.009 for the detached model: roughly twice the variance, not sixteen times. On
K562 there is no instability at all. We characterize this as a tail-risk failure mode on smaller,
sparser graphs rather than a general property, and we do not claim it transfers automatically to
other settings. Toggle the two lines above to reproduce the ablation.

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
| Matched-K μWD and frontier | `matched_k_mu_wd.py`, `mu_wd_frontier.py` |
| Held-out re-scoring and baselines | `rescore_ate.py`, `linear_baselines.py`, `reliability_check.py` |
| Directed-bridge collapse | `directed_bridge_sweep.py` |
| Biological modules | See [Biological analysis](#biological-analysis) |

> Note on config naming: `gnn_v4*` = the detached model reported in the paper; `gnn_v3*` = the
> no-detach ablation; `gnn_vae_*` are earlier runs retained for provenance and are **not** the
> paper's configuration.
> The five reported RPE1 seeds come from two run families: `gnn_v4sweep_*` supplies seeds 0–2
> and `gnn_v4_*` supplies seeds 3–4, as neither family contains all five. Both use identical
> configurations and the same detached encoder. The two families were trained four days apart;
> seeds present in both differ by ≈0.018 in ATE-ρ despite identical YAML and identical
> `pl.seed_everything()` calls, so a fresh reproduction from this tree will match seeds 0–2 and
> may not match seeds 3–4 exactly. Reported means use one run per seed.

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

> Because μWD depends strongly on edge count (see [Evaluation findings](#evaluation-findings)),
> the fact that our operating points are sparser than the baseline's is not incidental — it is
> what produces the μWD difference in the headline table. Compare at matched K.

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

**Threshold sensitivity.** Sweeping probability {0.60, 0.65, 0.70} × WD {0.2, 0.3, 0.4} ×
parent range over five K562 seeds shows the WD and parent-count criteria are non-binding at the
reported operating point: every edge above probability 0.6 already carries WD > 0.4, and every
retained anchor already has three or more parents. What is described as a triple criterion is in
effect a single one, edge probability. Subnetwork counts are threshold-sensitive (17.2, 9.2, 3.4
across the grid), but all five programs recover in all five seeds at 0.6. Programs, not anchors,
are the reproducible unit. Run `threshold_sweep.py`.

Related scripts: `subnetworks.py` and `subnetworks_g0.py` implement earlier and `G°`-only
variants of the subnetwork definition.

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

**2. The ATE reference is computed from all cells, and this is not changed here.**

`eval.py` calls `get_estimated_average_treatment_effects()` without a `split` argument, so the
reference average treatment effect every model is scored against is built from all cells —
including the 64% used for training. The reference is therefore not held out.

The `split` parameter is not an oversight in one file. It is declared on the abstract base class
(`gpo_vae/data/utils/perturbation_datamodule.py`) and implemented in all three concrete data
modules, where it applies a cell-level filter before the effect is estimated:

```python
if split is not None:
    adata = adata[adata.obs["split"] == split]
```

`eval.py` is the only call site for this method in the repository, and it does not pass it.

Relatedly, the `ATE_pearsonr-all`, `-train`, `-val` and `-test` keys in `test_metrics.csv` are
**bit-identical**. The loop that produces them (`for split in ["train", "val", "test"]`) selects
*perturbations*: it asks, for each perturbation, whether that perturbation appears in the given
split. Splits here are assigned by a random `train_test_split` over cell indices, so every
perturbation appears in all three and the filter removes nothing. Four names, one number.

**Where this comes from, and why it is reasonable code.** This evaluation path is inherited from
[CRADLE-VAE](https://github.com/dmis-lab/CRADLE-VAE), which shares it with GPO-VAE unchanged.
CRADLE-VAE also evaluates Norman and Dixit, and those datasets are split by a **perturbation-level
holdout** rather than at the cell level — `_get_split_labels` shuffles the combination guide
identities and assigns disjoint sets to train and test:

```python
train_combos = combo_guide_identities[:num_train_combos]
test_combos  = combo_guide_identities[-num_test_combos:]
```

Under that design the code is correct. A held-out combination's cells lie entirely within the test
split, so its reference effect is built from held-out data whether or not `split` is passed;
passing it would only shrink the cell count, and therefore the precision, for the seen
perturbations. The per-split ATE keys are likewise meaningful there, because a held-out
combination's row in `intervention_info` really is `(train=False, val=False, test=True)`.

What does not carry over is the assumption. GPO-VAE retains Adamson and Replogle, which use random
cell-level splits, and drops the two datasets where the perturbation-level machinery was doing
work. Nothing in the code signals the change: no assertion fires, and the only symptom is four
equal columns in a CSV.

**Why only the reference can be restricted.** The model side of this metric never reads observed
cells. `estimate_model_average_treatment_effect` receives only one-hot perturbation dosages, a
quality vector, `library_size=10000`, a particle count and a fixed seed; global parameters are
drawn from the trained guide and basal states from the model's generative prior. The model ATE is
therefore a deterministic function of the checkpoint. "Restrict the model to test cells" is not a
meaningful operation, and re-scoring requires no retraining — but `model_ate` is not written to
disk, so it must be regenerated per checkpoint.

**Scope of the effect.** This affects every number produced by this harness, ours and the
baseline's. Two mechanisms are involved and only one is contamination. The other is precision: the
all-cells reference uses roughly 5× more cells per perturbation, so it estimates the true effect
more cleanly, and any predictor scores higher against a cleaner reference. Our own numbers show
the precision term is large — test-vs-val (both held out, no model involved) gives ρ 0.515 on RPE1
against 0.661 for half-train-vs-half-train. We do not claim the two are separable with these data,
and we make no claim about how the correction would reorder baselines quoted from the upstream
paper, which we cannot run.

**We have deliberately not changed the default**, because the code as-is is what reproduces the
published values. To score against a held-out reference, use `rescore_ate.py`, which reports both.

**3. GRN metrics are `G°`-restricted.** μWD, FOR and edge counts are computed on the perturbed
block only (`adj_matrix[:pert_gene_len, :pert_gene_len]`, self-loops removed), for comparability
with causal-discovery baselines. The full-matrix count is reported separately as
`num_edges_full`. The `G⁺` block is saturated (~800k edges > 0.5) in **both** models and is not
a meaningful quality signal.

**4. μWD is `positive_mean_wasserstein`.** `output_graph['wasserstein_distance']['mean']`.
`negative_mean_wasserstein` is a different quantity. See
[Evaluation findings](#evaluation-findings) before interpreting it as a quality measure.

**5. Lightning directory auto-increment.** Re-running a config whose `name` already exists
creates `<name>-2`, `<name>-3`, etc. rather than overwriting. Verify you are evaluating the run
you think you are; `grn.csv` should have 1547 columns for K562 (1546 genes + index).

**6. Run eval with `WANDB_MODE=disabled`** unless you have configured wandb.

**7. `qc_pass` is not forwarded on the local-directory path.** `evaluate_local_experiment`
accepts `qc_pass` but does not pass it to `evaluate_checkpoint`, while
`evaluate_local_checkpoint` and `evaluate_wandb_experiment` both do. Local-directory evaluations
therefore run at the default regardless of the `--qc_pass` flag. All numbers reported here were
produced with `qc_pass=False` throughout, which is what reproduces the published baseline.

---

## Held-out re-scoring and non-parametric baselines

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

Results from all three are tabulated under [Evaluation findings](#evaluation-findings).

All three use the `perturbseq` scale (normalize to 1e4 UMI, `log2(x+1)`, *then* difference
against the control mean) — the same scale `eval.py --perturbseq` uses. The `mean` scale
differences raw counts instead and gives substantially different values; on the `mean` scale the
per-perturbation baseline scores 0.850 on RPE1 against 0.658 on `perturbseq`, because a handful of
high-expression genes carry the correlation. Always use `perturbseq` for comparisons against
published numbers.

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
├── rescore_ate.py                          # held-out ATE re-scoring
├── linear_baselines.py                     # non-parametric baselines
├── reliability_check.py                    # measurement-ceiling estimate
├── matched_k_mu_wd.py                      # μWD at matched edge count
├── mu_wd_frontier.py                       # μWD-vs-edge-count frontier
├── extract_edge_wd.py                      # per-edge Wasserstein distances
├── subnetworks_v2.py                       # convergence subnetworks
├── enrich_subnets_bg.py                    # KEGG/Reactome ORA
├── build_handoff.py                        # per-subnetwork report
├── aggregate_subnets.py                    # cross-seed module recovery
├── directed_bridge_sweep.py                # directed-bridge collapse analysis
├── threshold_sweep.py                      # convergence-threshold sensitivity
├── profile_memory.py / profile_eval.py     # memory and runtime profiling
├── aggregate_*.py                          # results aggregation
└── run_*.sh                                # multi-GPU job schedulers
```

---

## Citation

This manuscript is under review. A citation will be added here upon acceptance.

Please cite the base method:

```bibtex
@article{gpovae2025,
  title   = {{GPO-VAE}: modeling explainable gene perturbation responses
             utilizing {GRN}-aligned parameter optimization},
  author  = {Baek, Seungheun and Park, Soyon and Chok, Yan Ting and
             Gim, Mogan and Kang, Jaewoo},
  journal = {Bioinformatics},
  volume  = {41},
  number  = {Supplement\_1},
  pages   = {i599--i608},
  year    = {2025},
  doi     = {10.1093/bioinformatics/btaf256}
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
