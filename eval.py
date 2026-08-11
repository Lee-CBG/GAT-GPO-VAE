"""
Script to generate test set evaluation metrics from a training run
Usage:
    python eval.py --experiment_path results/example --perturbseq
    - Saves evaluation metrics to results/example/test_metrics.csv
"""
import argparse
import os
import sys
from os.path import basename, join, splitext
from typing import Any, Dict, Literal

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

os.chdir('/'.join(__file__.split('/')[:-1]))
sys.path.append('/'.join(__file__.split('/')[:-1]))

from gpo_vae.data.utils.anndata import align_adatas
from gpo_vae.models.utils.perturbation_lightning_module import (
    TrainConfigPerturbationLightningModule,
)
import cupy as cp


def evaluate_checkpoint(
    checkpoint_path: str,
    average_treatment_effect_method: Literal["mean", "perturbseq"],
    batch_size: int = 500,
    ate_n_particles: int = 2500,
    qc_pass: bool = False,
    thr: int = 5,
    max_path_length: int = 3,
    omission_estimation_size: int = 500,
    p_value_threshold: float = 0.05,
    devices=None,
) -> Dict[str, Any]:
    """
    Compute test set metrics for a given checkpoint.
    Evaluation follows the paper's methodology:
    - ATE metrics over all perturbations. NOTE: the reference ATE is computed
      from ALL cells, including those used for training, because
      get_estimated_average_treatment_effects() is called below without a
      `split` argument. Inherited from upstream GPO-VAE; left unchanged so that
      published numbers reproduce. The ATE_pearsonr-{all,train,val,test} keys
      are therefore bit-identical (the split applied here is by perturbation,
      and every perturbation appears in every split). See README,
      Reproducibility note 2, and rescore_ate.py for a leakage-free reference.
    - GRN metrics over G° only (perturbed genes, no self-loops)
    - Statistical evaluation (network/WD/FOR metrics) uses TEST-SPLIT cells only
      (predictor.py ~L465)
    """
    lightning_module = load_checkpoint(checkpoint_path, devices)
    data_module = lightning_module.get_data_module()
    thr = data_module.thr_values
    predictor = lightning_module.predictor
    metrics = {}

    # IWELBO
    test_loader = DataLoader(
        data_module.test_dataloader().dataset,
        batch_size=batch_size,
    )
    test_iwelbo_df = predictor.compute_predictive_iwelbo(
        loaders=test_loader, n_particles=100
    )
    metrics["test/IWELBO"] = test_iwelbo_df["IWELBO"].mean()

    # ATE metrics
    data_ate = data_module.get_estimated_average_treatment_effects(
        method=average_treatment_effect_method,
        qc_pass=qc_pass,
    )
    if data_ate is not None:
        model_ate = predictor.estimate_average_effects_data_module(
            data_module=data_module,
            control_label=data_ate.uns["control"],
            method=average_treatment_effect_method,
            n_particles=ate_n_particles,
            condition_values=dict(library_size=10000 * torch.ones((1,))),
            batch_size=batch_size,
        )
        data_ate, model_ate = align_adatas(data_ate, model_ate)
        intervention_info = data_module.get_unique_observed_intervention_info()
        metrics["ATE_n_particles"] = ate_n_particles

        ate_metrics_all = get_ate_metrics(data_ate, model_ate)
        for k, v in ate_metrics_all.items():
            metrics[f"{k}-all"] = v

        for split in ["train", "val", "test"]:
            split_perts = intervention_info[intervention_info[split]].index
            idx = data_ate.obs.index.isin(split_perts)
            ate_metrics_split = get_ate_metrics(data_ate[idx], model_ate[idx])
            for k, v in ate_metrics_split.items():
                metrics[f"{k}-{split}"] = v

    # GRN evaluation
    print('Start Network Statistical Evaluation')
    guide_dists, guide_samples = predictor.guide()
    grn_dist = guide_dists["q_mask"]
    grn = grn_dist.probs.detach().cpu()

    d_var_info = data_module.d_var_info.index
    pd.DataFrame(grn, columns=d_var_info, index=d_var_info).to_csv(
        f"{'/'.join(checkpoint_path.split('/')[:-2])}/grn.csv"
    )

    adj_matrix = (grn > 0.5).type(torch.FloatTensor)

    # Restrict to G° only — paper Section 3.2:
    # "we exclude the extended genes G+ and include only the perturbed genes G°"
    pert_gene_len = len(data_module.adata.obs['gene'].unique()) - 1
    adj_matrix_pert = adj_matrix[:pert_gene_len, :pert_gene_len]

    # Remove self-loops
    mask_no_diag = ~torch.eye(pert_gene_len, dtype=torch.bool)
    adj_matrix_pert = adj_matrix_pert * mask_no_diag.float()

    # ---- G° edge list (restricted, no self-loops) ----
    edge_list_g0 = [
        (i.item(), j.item())
        for i, j in zip(*torch.nonzero(adj_matrix_pert, as_tuple=True))
    ]

    # ---- Full matrix edge list (paper-original, no restriction) ----
    edge_list_full = [
        (i.item(), j.item())
        for i, j in zip(*torch.nonzero(adj_matrix, as_tuple=True))
    ]

    metrics["num_edges"] = adj_matrix_pert.sum().item()
    metrics["num_edges_full"] = adj_matrix.sum().item()

    for tag, edge_list in [("g0", edge_list_g0), ("full", edge_list_full)]:
        qte = predictor.statistical_evaluation(
            network=edge_list,
            data_module=data_module,
            n_particles=100,
            condition_values=dict(library_size=10000 * torch.ones((1,))),
            max_path_length=max_path_length,
            check_false_omission_rate=True,
            omission_estimation_size=omission_estimation_size,
            p_value_threshold=p_value_threshold,
        )
        for k, v in qte.items():
            metrics[f"{k}__{tag}"] = v
        if "output_graph" in qte:
            og = qte["output_graph"]
            if isinstance(og, dict) and "wasserstein_distance" in og:
                metrics[f"positive_mean_wasserstein__{tag}"] = og["wasserstein_distance"]["mean"]

    # Back-compat: keep unsuffixed keys pointing at G° (current default)
    metrics["false_omission_rate"] = metrics.get("false_omission_rate__g0")
    metrics["positive_mean_wasserstein"] = metrics.get("positive_mean_wasserstein__g0")

    return metrics


def get_ate_metrics(data_ate, model_ate):
    metrics = {}

    top_20_idx_X = np.argpartition(np.abs(data_ate.X.copy()), data_ate.shape[1] - 20)[:, -20:]
    top_20_idx_Y = np.argpartition(np.abs(model_ate.X.copy()), data_ate.shape[1] - 20)[:, -20:]
    top_50_idx_X = np.argpartition(np.abs(data_ate.X.copy()), data_ate.shape[1] - 50)[:, -50:]
    top_50_idx_Y = np.argpartition(np.abs(model_ate.X.copy()), data_ate.shape[1] - 50)[:, -50:]

    x = data_ate.X.flatten()
    y = model_ate.X.flatten()
    metrics["ATE_pearsonr"] = pearsonr(x, y)[0]
    metrics["ATE_r2"] = r2_score(x, y)

    x = np.take_along_axis(data_ate.X.copy(), top_20_idx_X, axis=-1).flatten()
    y = np.take_along_axis(model_ate.X.copy(), top_20_idx_X, axis=-1).flatten()
    metrics["ATE_pearsonr_top20"] = pearsonr(x, y)[0]
    metrics["ATE_r2_top20"] = r2_score(x, y)

    metrics["jaccard_sim_top20"] = jaccard_sim(top_20_idx_X, top_20_idx_Y)
    metrics["jaccard_sim_top50"] = jaccard_sim(top_50_idx_X, top_50_idx_Y)

    return metrics


def jaccard_sim(X, Y):
    return np.mean([
        len(set(x) & set(y)) / len(set(x) | set(y))
        for x, y in zip(X, Y)
    ])


def evaluate_local_experiment(
    experiment_path: str,
    average_treatment_effect_method: Literal["mean", "perturbseq"],
    batch_size: int = 128,
    ate_n_particles: int = 2500,
    qc_pass: bool = False,
    thr: int = 5,
    devices=None,
    bioeval_dir_path: str = None,
):
    checkpoint_names = os.listdir(join(experiment_path, "checkpoints"))
    best_checkpoints = [x for x in checkpoint_names if x[:4] == "best"]
    assert len(best_checkpoints) == 1
    checkpoint_path = join(experiment_path, "checkpoints", best_checkpoints[0])
    checkpoint_name = splitext(basename(checkpoint_path))[0]

    metrics = evaluate_checkpoint(
        checkpoint_path,
        average_treatment_effect_method=average_treatment_effect_method,
        batch_size=batch_size,
        ate_n_particles=ate_n_particles,
        thr=thr,
    )
    metrics["checkpoint"] = checkpoint_name
    metrics_df = pd.DataFrame({k: [v] for k, v in metrics.items()}).T
    metrics_path = join(experiment_path, "test_metrics.csv")
    metrics_df.to_csv(metrics_path)
    print(f"\nMetrics saved to {metrics_path}")
    print(metrics_df)


def evaluate_local_checkpoint(
    checkpoint_path: str,
    average_treatment_effect_method: Literal["mean", "perturbseq"],
    batch_size: int = 128,
    ate_n_particles: int = 2500,
    qc_pass: bool = False,
    thr: int = 5,
    devices=None,
    bioeval_dir_path: str = None,
):
    checkpoint_base = splitext(checkpoint_path)[0]
    checkpoint_name = splitext(basename(checkpoint_path))[0]

    metrics = evaluate_checkpoint(
        checkpoint_path,
        average_treatment_effect_method=average_treatment_effect_method,
        batch_size=batch_size,
        ate_n_particles=ate_n_particles,
        qc_pass=qc_pass,
        thr=thr,
    )
    metrics["checkpoint"] = checkpoint_name
    metrics_df = pd.DataFrame({k: [v] for k, v in metrics.items()}).T
    metrics_path = checkpoint_base + "_test_metrics.csv"
    metrics_df.to_csv(metrics_path)
    print(f"\nMetrics saved to {metrics_path}")
    print(metrics_df)


def load_checkpoint(checkpoint_path: str, devices):
    if devices is None:
        return TrainConfigPerturbationLightningModule.load_from_checkpoint(checkpoint_path)
    return TrainConfigPerturbationLightningModule.load_from_checkpoint(
        checkpoint_path,
        map_location=lambda storage, loc: storage.cuda(devices) if torch.cuda.is_available() else storage,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_path", default="results/gpo_vae_rpe1")
    parser.add_argument("--perturbseq", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--ate_n_particles", type=int, default=2500)
    parser.add_argument("--qc_pass", type=bool, default=False)
    parser.add_argument("--devices", type=int, default=0)
    parser.add_argument("--thr", type=int, default=3)

    args = parser.parse_args()
    cp.cuda.Device(args.devices).use()
    method: Literal["mean", "perturbseq"] = "perturbseq" if args.perturbseq else "mean"

    if os.path.isdir(args.experiment_path):
        evaluate_local_experiment(
            args.experiment_path,
            method,
            batch_size=args.batch_size,
            ate_n_particles=args.ate_n_particles,
            thr=args.thr,
            devices=args.devices,
        )
    else:
        evaluate_local_checkpoint(
            args.experiment_path,
            method,
            batch_size=args.batch_size,
            ate_n_particles=args.ate_n_particles,
            qc_pass=args.qc_pass,
            thr=args.thr,
            devices=args.devices,
        )