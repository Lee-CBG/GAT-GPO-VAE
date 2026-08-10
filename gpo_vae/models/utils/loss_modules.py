from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import expm, norm


class PerturbationPlatedELBOLossModule(torch.nn.Module):
    """
    Computes ELBO with option to specify variables that are on perturbation-based
    plates (eg perturbation embeddings that are shared between samples that receive
    a given perturbation)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        guide: torch.nn.Module,
        local_variables: Optional[Iterable[str]] = None,
        perturbation_plated_variables: Optional[Iterable[str]] = None,
        pos_weight: int = None,
        alpha: int = None,
        beta: int = 1,
        hop: int = 0,
        knowledge_path: str = None,
        fc_criteria: int = 2,
        gloss_coeff: int = 100,
        penaly_coeff: int = 1,
    ):
        super().__init__()
        local_variables = list(local_variables) if local_variables is not None else []
        perturbation_plated_variables = (
            list(perturbation_plated_variables)
            if perturbation_plated_variables is not None
            else []
        )

        assert sorted(model.get_var_keys()) == sorted(
            guide.get_var_keys()
        ), "Mismatch in model and guide variables"

        variables = local_variables + perturbation_plated_variables
        assert sorted(list(model.get_var_keys())) == sorted(
            variables
        ), "Mismatch between model variables and variables specified to loss module"

        self.model = model
        self.guide = guide
        self.local_variables = local_variables
        self.perturbation_plated_variables = perturbation_plated_variables
        self.pos_weight = pos_weight
        self.alpha = alpha
        self.beta = beta
        self.hop = hop
        self.fc_criteria = fc_criteria
        self.gloss_coeff = gloss_coeff
        self.penaly_coeff = penaly_coeff
        if knowledge_path is not None:
            self.prior_network = torch.from_numpy(np.load(knowledge_path)).float()

    def forward(
        self,
        X: torch.Tensor,
        D: torch.Tensor,
        qc: torch.Tensor,
        cfX: torch.Tensor,
        condition_values: Optional[Dict[str, torch.Tensor]] = None,
        n_particles: int = 1,
    ):
        if condition_values is None:
            condition_values = dict()

        pred_qc = None

        guide_dists, guide_samples = self.guide(
            X=X,
            D=D,
            qc=qc,
            n_particles=n_particles,
        )

        for k, v in guide_samples.items():
            condition_values[k] = v

        model_dists, model_samples = self.model(
            D=D,
            qc=qc,
            condition_values=condition_values,
            n_particles=n_particles,
        )

        cf_guide_dists, cf_guide_samples = self.guide(
            X=cfX,
            D=D,
            qc=(1 - qc),
            n_particles=n_particles,
        )

        return guide_dists, model_dists, model_samples, cf_guide_dists, cf_guide_samples, pred_qc

    def loss(
        self,
        X: torch.Tensor,
        D: torch.Tensor,
        D_obs_counts: torch.Tensor,
        qc: torch.Tensor,
        qc_obs_counts: torch.Tensor,
        deltaX: torch.Tensor = None,
        cfX: list = None,
        condition_values: Optional[Dict[str, torch.Tensor]] = None,
        n_particles: int = 1,
        test: bool = False,
        x_var_info=None,
    ):
        guide_dists, model_dists, samples, cf_guide_dists, cf_guide_samples, pred_qc = self.forward(
            X=X,
            D=D,
            cfX=cfX,
            qc=qc,
            condition_values=condition_values,
            n_particles=n_particles,
        )

        loss_terms = {}

        # ── Reconstruction loss ────────────────────────────────────────────────
        p_x_good_log_p = model_dists['p_x_good'].log_prob(X)
        p_x_bad_log_p = model_dists['p_x_bad'].log_prob(X)
        alpha = self.alpha
        if alpha is None:
            alpha = qc.sum() / qc.shape[0]
        if qc.shape[1] > 1:
            qc = qc.max(axis=1)[0].unsqueeze(-1)
        p_x_real_log_p = alpha * ((1 - qc) * p_x_good_log_p) + (1 - alpha) * (qc * p_x_bad_log_p)
        loss_terms["reconstruction"] = p_x_real_log_p.sum(-1)

        # ── Counterfactual KL loss ─────────────────────────────────────────────
        idx = torch.nonzero(cfX.sum(1) != 0).squeeze(1)
        kl_loss = self.beta * self.kldiv_normal(
            cf_guide_dists['q_z_basal'].mean[:, idx, :],
            cf_guide_dists['q_z_basal'].stddev[:, idx, :],
            guide_dists['q_z_cf_basal'].mean[:, idx, :],
            guide_dists['q_z_cf_basal'].stddev[:, idx, :],
        )
        kl_loss = kl_loss.sum()

        # ── GRN loss ───────────────────────────────────────────────────────────
        # if "H_L" in guide_dists:
        #     # GNN model: use graph-informed node embeddings
        #     # H_L shape: [n_particles × n_genes × n_genes]
        #     H_L = guide_dists["H_L"].mean(0)  # [n_genes × n_genes]
        #     grn_loss = self.gloss_coeff * F.l1_loss(D @ H_L, deltaX - self.fc_criteria)
        # else:
        #     # Original GPO-VAE: matrix power accumulation (T̂_K)
        #     A = guide_dists['q_mask'].logits
        #     S = A.clone()
        #     current_term = A.clone()
        #     for _ in range(self.hop):
        #         current_term = (current_term @ A) / A.shape[0]
        #         S += current_term
        #     grn_loss = self.gloss_coeff * F.l1_loss(D @ S, deltaX - self.fc_criteria)
        # AFTER (Option B — always use T̂_K on Ŵ logits, regardless of GNN or not):

        A = guide_dists['q_mask'].logits
        S = A.clone()
        current_term = A.clone()
        for _ in range(self.hop):
            current_term = (current_term @ A) / A.shape[0]
            S += current_term
        grn_loss = self.gloss_coeff * F.l1_loss(D @ S, deltaX - self.fc_criteria)
        # Note: H_L (GAT output) is still used for E ~ N(H_L, sigma_E) in the guide,
        # but GRN loss now goes directly through Ŵ logits as in original GPO-VAE.

        # ── Sparsity penalty ───────────────────────────────────────────────────
        n_genes = guide_dists['q_mask'].probs.shape[0]
        if self.penaly_coeff > 0:
            penalty = self.penaly_coeff * self.compute_penalty(
                [guide_dists['q_mask'].probs], p=1
            )
            penalty = penalty / (n_genes ** 2)
        else:
            penalty = torch.tensor(0.0, device=X.device)

        # ── KL regularization for all latent variables ─────────────────────────
        # Skip non-distribution keys: H_L, q_z_cf_basal
        skip_keys = {'q_z_cf_basal', 'H_L'}
        for k in guide_dists.keys():
            if k in skip_keys:
                continue
            # also skip any key that doesn't have a matching prior
            var_key = k[2:]  # drop 'q_' prefix
            prior_key = f"p_{var_key}"
            if prior_key not in model_dists:
                continue
            if var_key not in samples:
                continue

            loss_term = model_dists[prior_key].log_prob(samples[var_key])
            loss_term = loss_term - guide_dists[k].log_prob(samples[var_key])

            if var_key in self.perturbation_plated_variables:
                if 'qc' in var_key:
                    loss_term = self._compute_reweighted_perturbation_plated_loss_term(
                        qc, qc_obs_counts, loss_term
                    )
                else:
                    loss_term = self._compute_reweighted_perturbation_plated_loss_term(
                        D, D_obs_counts, loss_term
                    )

            loss_term = loss_term.sum(-1)
            loss_terms[var_key] = loss_term

        # ── Total loss ─────────────────────────────────────────────────────────
        batch_elbo: torch.Tensor = sum([v for k, v in loss_terms.items()])
        loss = -batch_elbo.mean()
        loss += kl_loss
        loss += grn_loss
        loss += penalty

        metrics = {
            f"loss_term_{k}": -v.detach().cpu().mean() for k, v in loss_terms.items()
        }
        metrics["cf_KLdivergence"] = kl_loss.detach().cpu()
        metrics["grn_loss"] = grn_loss.detach().cpu()
        metrics["penalty"] = penalty.detach().cpu()

        return loss, metrics

    def kldiv_normal(
        self,
        mu1: torch.Tensor,
        sigma1: torch.Tensor,
        mu2: torch.Tensor,
        sigma2: torch.Tensor,
    ) -> torch.Tensor:
        logvar1 = 2 * sigma1.log()
        logvar2 = 2 * sigma2.log()
        return torch.mean(
            -0.5 * torch.sum(
                1.0 + logvar1 - logvar2
                - (mu1 - mu2) ** 2
                - (logvar1 - logvar2).exp(),
                dim=1,
            ),
            dim=0,
        )

    def compute_dag_constraint(self, w_adj):
        h = TrExpScipy.apply(w_adj) - w_adj.shape[0]
        return h

    def spectral_radius_regularization_loss(self, A, max_radius=1.0, num_iters=50):
        assert A.size(0) == A.size(1)
        b_k = torch.rand(A.size(0), 1, device=A.device)
        for _ in range(num_iters):
            b_k1 = torch.matmul(A, b_k)
            b_k1_norm = torch.norm(b_k1, p=2)
            b_k = b_k1 / b_k1_norm
        spectral_radius = torch.norm(torch.matmul(A, b_k), p=2) / torch.norm(b_k, p=2)
        return torch.relu(spectral_radius - max_radius)

    def cosine_similarity_loss(self, tensor1, tensor2):
        return 1 - F.cosine_similarity(tensor1.view(1, -1), tensor2.view(1, -1)).mean()

    def deltaX_reconstruction(self, loc, scale, deltaX, D):
        deltaX_expanded = deltaX.unsqueeze(0)
        D_expanded = D.unsqueeze(0)
        loc_selected = D_expanded @ loc.unsqueeze(0)
        scale_selected = D_expanded @ scale.unsqueeze(0)
        selected_dist = torch.distributions.Normal(
            loc_selected, torch.clamp(scale_selected, min=1e-5)
        )
        grn_log_probs = selected_dist.log_prob(deltaX_expanded)
        return -grn_log_probs.sum(-1).mean()

    def normalize_min_max(self, matrix):
        min_vals = torch.min(matrix, axis=0, keepdims=True).values
        max_vals = torch.max(matrix, axis=0, keepdims=True).values
        return (matrix - min_vals) / (max_vals - min_vals + 1e-8)

    def compute_penalty(self, list_, p=2, target=0.0):
        penalty = 0
        for m in list_:
            penalty += torch.norm(m - target, p=p) ** p
        return penalty

    def _compute_reweighted_perturbation_plated_loss_term(
        self, conditioning_variable, total_obs_per_condition, loss_term
    ):
        condition_nonzero = (conditioning_variable != 0).type(torch.float32)
        obs_scaling = 1 / total_obs_per_condition
        obs_scaling[torch.isinf(obs_scaling)] = 0
        obs_scaling = obs_scaling.reshape(1, -1)
        rw_condition_nonzero = condition_nonzero * obs_scaling
        rw_loss_term = torch.matmul(rw_condition_nonzero, loss_term)
        return rw_loss_term


class PerturbationPlatedELBOCustomReweightedLossModule(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        guide: torch.nn.Module,
        local_variables: Optional[Iterable[str]] = None,
        perturbation_plated_variables: Optional[Iterable[str]] = None,
        custom_prior_weights: Optional[Dict[str, float]] = None,
        custom_plated_prior_additional_weight_proportional_n: Optional[Dict[str, float]] = None,
        custom_loss_term_weights: Optional[Iterable[str]] = None,
        custom_plated_loss_term_additional_weight_proportional_n: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        local_variables = list(local_variables) if local_variables is not None else []
        perturbation_plated_variables = (
            list(perturbation_plated_variables)
            if perturbation_plated_variables is not None
            else []
        )
        assert sorted(model.get_var_keys()) == sorted(guide.get_var_keys())
        variables = local_variables + perturbation_plated_variables
        assert sorted(list(model.get_var_keys())) == sorted(variables)

        self.model = model
        self.guide = guide
        self.local_variables = local_variables
        self.perturbation_plated_variables = perturbation_plated_variables

        if custom_prior_weights is not None:
            assert set(custom_prior_weights.keys()).issubset(set(variables))
        if custom_plated_prior_additional_weight_proportional_n is not None:
            assert set(custom_plated_prior_additional_weight_proportional_n.keys()).issubset(
                set(perturbation_plated_variables)
            )
        if custom_loss_term_weights is not None:
            assert set(custom_loss_term_weights).issubset(set(variables))
        if custom_plated_loss_term_additional_weight_proportional_n is not None:
            assert set(custom_plated_loss_term_additional_weight_proportional_n.keys()).issubset(
                set(perturbation_plated_variables)
            )

        custom_weights_dicts = [
            custom_loss_term_weights,
            custom_plated_loss_term_additional_weight_proportional_n,
            custom_prior_weights,
            custom_plated_prior_additional_weight_proportional_n,
        ]
        if sum(d is not None for d in custom_weights_dicts) > 1:
            raise NotImplementedError("Handling multiple reweighting types not implemented")

        self.custom_prior_weights = custom_prior_weights
        self.custom_plated_prior_additional_weight_proportional_n = custom_plated_prior_additional_weight_proportional_n
        self.custom_loss_term_weights = custom_loss_term_weights
        self.custom_plated_loss_term_additional_weight_proportional_n = custom_plated_loss_term_additional_weight_proportional_n

    def forward(self, X, D, qc, qc_obs_counts, condition_values=None, n_particles=1):
        if condition_values is None:
            condition_values = dict()
        guide_dists, guide_samples = self.guide(X=X, D=D, qc=qc, n_particles=n_particles)
        for k, v in guide_samples.items():
            condition_values[k] = v
        model_dists, model_samples = self.model(D=D, qc=qc, condition_values=condition_values, n_particles=n_particles)
        return guide_dists, model_dists, model_samples

    def loss(self, X, D, D_obs_counts, qc, qc_obs_counts, condition_values=None, n_particles=1):
        guide_dists, model_dists, samples = self.forward(
            X=X, D=D, qc=qc, condition_values=condition_values, n_particles=n_particles
        )
        loss_terms = {}
        loss_terms["reconstruction"] = model_dists["p_x"].log_prob(X).sum(-1)
        for k in guide_dists.keys():
            var_key = k[2:]
            p = model_dists[f"p_{var_key}"].log_prob(samples[var_key])
            q = guide_dists[f"q_{var_key}"].log_prob(samples[var_key])
            if var_key in self.perturbation_plated_variables:
                p = self._compute_reweighted_perturbation_plated_loss_term(
                    var_key, D, D_obs_counts, qc, qc_obs_counts, p, is_prior=True
                )
                q = self._compute_reweighted_perturbation_plated_loss_term(
                    var_key, D, D_obs_counts, qc, qc_obs_counts, q, is_prior=False
                )
            if self.custom_prior_weights is not None:
                if var_key in self.custom_prior_weights:
                    p = self.custom_prior_weights[var_key] * p
            if self.custom_loss_term_weights is not None:
                if var_key in self.custom_loss_term_weights:
                    p = self.custom_loss_term_weights[var_key] * p
                    q = self.custom_loss_term_weights[var_key] * q
            loss_term = p - q
            loss_term = loss_term.sum(-1)
            loss_terms[var_key] = loss_term
        batch_elbo: torch.Tensor = sum([v for k, v in loss_terms.items()])
        loss = -batch_elbo.mean()
        metrics = {f"loss_term_{k}": -v.detach().cpu().mean() for k, v in loss_terms.items()}
        return loss, metrics

    def _compute_reweighted_perturbation_plated_loss_term(
        self, var_key, conditioning_variable, total_obs_per_condition, loss_term, is_prior=False
    ):
        condition_nonzero = (conditioning_variable != 0).type(torch.float32)
        obs_scaling = 1 / total_obs_per_condition
        obs_scaling[torch.isinf(obs_scaling)] = 0
        if is_prior and self.custom_plated_prior_additional_weight_proportional_n is not None:
            if var_key in self.custom_plated_prior_additional_weight_proportional_n:
                obs_scaling = obs_scaling + self.custom_plated_prior_additional_weight_proportional_n[var_key]
        if self.custom_plated_loss_term_additional_weight_proportional_n is not None:
            if var_key in self.custom_plated_loss_term_additional_weight_proportional_n:
                obs_scaling = obs_scaling + self.custom_plated_loss_term_additional_weight_proportional_n[var_key]
        obs_scaling = obs_scaling.reshape(1, -1)
        rw_condition_nonzero = condition_nonzero * obs_scaling
        return torch.matmul(rw_condition_nonzero, loss_term)


class PerturbationPlatedIWELBOLossModule(PerturbationPlatedELBOLossModule):
    def loss(self, X, D, D_obs_counts, qc, qc_obs_counts, condition_values=None, n_particles=1):
        guide_dists, model_dists, samples = self.forward(
            X=X, D=D, qc=qc, condition_values=condition_values, n_particles=n_particles
        )
        loss_terms = {}
        loss_terms["reconstruction"] = model_dists["p_x"].log_prob(X).sum(-1)
        for k in guide_dists.keys():
            var_key = k[2:]
            loss_term = model_dists[f"p_{var_key}"].log_prob(samples[var_key])
            loss_term = loss_term - guide_dists[f"q_{var_key}"].log_prob(samples[var_key])
            if var_key in self.perturbation_plated_variables:
                if 'qc' in var_key:
                    loss_term = self._compute_reweighted_perturbation_plated_loss_term(
                        qc, qc_obs_counts, loss_term
                    )
                else:
                    loss_term = self._compute_reweighted_perturbation_plated_loss_term(
                        D, D_obs_counts, loss_term
                    )
            loss_term = loss_term.sum(-1)
            loss_terms[var_key] = loss_term
        loss_terms_tensor = torch.stack([v for k, v in loss_terms.items()]).sum(0)
        batch_iwelbo = torch.logsumexp(loss_terms_tensor, dim=0)
        batch_iwelbo = batch_iwelbo - np.log(n_particles)
        loss = -batch_iwelbo.mean()
        metrics = {f"loss_term_{k}": -v.detach().cpu().mean() for k, v in loss_terms.items()}
        with torch.no_grad():
            batch_elbo: torch.Tensor = sum([v for k, v in loss_terms.items()])
            metrics["batch_elbo"] = batch_elbo.mean()
        return loss, metrics


class TrExpScipy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        with torch.no_grad():
            input_np = input.detach().cpu().numpy()
            expm_input = expm(input_np / norm(input_np))
            expm_input = torch.as_tensor(expm_input)
            if input.is_cuda:
                expm_input = expm_input.to(input.device)
            ctx.save_for_backward(expm_input)
            return torch.trace(expm_input)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            (expm_input,) = ctx.saved_tensors
            return expm_input.t() * grad_output