
"""
GNN-augmented guide for GPO-VAE.

Replaces the row-wise independent embedding encoder (which maps each gene's
mask row independently through an MLP) with a Graph Attention Network (GAT)
that does message passing over the gene regulatory graph defined by q_mask.

The key change vs the original correlated_normal_guide:
  - Original: each gene's embedding is computed independently from its own mask row
  - GNN:      each gene's embedding is informed by its regulatory neighbors via attention

Everything else (basal encoder, artifact encoder, Gumbel-Softmax mask sampling,
decoder, loss) is unchanged.

v2 changes vs v1:
  - Removed .detach() from edge_weight: gradient now flows GRN loss → H_L → GAT → Ŵ
  - Vectorized particle dimension: all P particles processed in one batched GAT call
  - Replaced hard edge mask with soft log-bias: no discontinuous graph changes
  - MEMORY FIX: split attention vector (a_src, a_dst) to avoid O(B*n²*H*2d) concat.
    Peak allocation is now the attention map [B,n,n,H] = 30MB for n=655, not 81GB.
"""

from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.utils import probs_to_logits

from gpo_vae.data.utils.perturbation_datamodule import ObservationNormalizationStatistics
from gpo_vae.models.utils.gumbel_softmax_bernoulli import GumbelSoftmaxBernoulliStraightThrough
from gpo_vae.models.utils.mlp import get_likelihood_mlp, qc_model
from gpo_vae.models.utils.normalization import get_normalization_module


# ──────────────────────────────────────────────────────────────────────────────
# Minimal GAT implementation (no PyG dependency)
# ──────────────────────────────────────────────────────────────────────────────

class GATLayer(nn.Module):
    """
    Single Graph Attention layer — vectorized over a batch dimension.

    Accepts H of shape [B, n_nodes, in_features] and edge_weight of shape
    [n_nodes, n_nodes] (shared across the batch), returns [B, n_nodes, out_features].

    Attention score (memory-efficient split formulation):
        e_ij = LeakyReLU(a_src^T Wh_i + a_dst^T Wh_j) + log(clamp(w_ij, 1e-6))

    Splitting a into a_src and a_dst avoids materializing [B, n, n, H, 2d].
    Peak allocation is the attention map [B, n, n, H] = 30MB for n=655, P=5.

    The log(w_ij) bias means:
      - Near-zero edges get very negative logits → ~0 attention after softmax
      - No hard threshold, no -inf discontinuities during training
      - Gradient flows through sigmoid(Ŵ) back into Ŵ (no detach)
    """

    def __init__(self, in_features: int, out_features: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert out_features % n_heads == 0, "out_features must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = out_features // n_heads
        self.out_features = out_features

        self.W = nn.Linear(in_features, out_features, bias=False)

        # Split attention vectors: a = [a_src || a_dst], each [n_heads, d_head]
        # This avoids forming [B, n, n, H, 2d] — saves ~80GB for n=655
        self.a_src = nn.Parameter(torch.empty(n_heads, self.d_head))
        self.a_dst = nn.Parameter(torch.empty(n_heads, self.d_head))
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(
        self,
        H: torch.Tensor,           # [B, n, in_features]
        edge_weight: torch.Tensor, # [n, n]  soft adjacency sigmoid(Ŵ), NO detach
    ) -> torch.Tensor:             # [B, n, out_features]
        B, n, _ = H.shape

        # Linear transform → [B, n, n_heads, d_head]
        Wh = self.W(H).view(B, n, self.n_heads, self.d_head)

        # Per-node attention scores — O(B*n*H*d), not O(B*n²*H*d)
        e_src = (Wh * self.a_src).sum(-1)  # [B, n, H]
        e_dst = (Wh * self.a_dst).sum(-1)  # [B, n, H]

        # e_ij = LeakyReLU(e_src_i + e_dst_j)
        # [B, n, 1, H] + [B, 1, n, H] → [B, n, n, H]  (only allocation of O(n²))
        e = self.leaky_relu(e_src.unsqueeze(2) + e_dst.unsqueeze(1))

        # Log-bias from soft adjacency — no detach, gradient flows into Ŵ
        # edge_weight: [n, n] → [1, n, n, 1]
        w = edge_weight[None, :, :, None]
        e = e + torch.log(w.clamp(min=1e-6))  # [B, n, n, H]

        # Softmax over source nodes (j = dim 2)
        alpha = torch.softmax(e, dim=2)        # [B, n, n, H]
        alpha = self.dropout(alpha)

        # Aggregate: out[b,i] = sum_j alpha[b,i,j] * Wh[b,j]  via bmm
        Wh_t = Wh.permute(0, 2, 1, 3)         # [B, H, n, d]
        alpha_t = alpha.permute(0, 3, 1, 2)    # [B, H, n, n]
        out = torch.bmm(
            alpha_t.reshape(B * self.n_heads, n, n),
            Wh_t.reshape(B * self.n_heads, n, self.d_head),
        ).reshape(B, self.n_heads, n, self.d_head)
        out = out.permute(0, 2, 1, 3).reshape(B, n, self.out_features)  # [B, n, out]
        return out


class GNNEmbeddingEncoder(nn.Module):
    """
    Replaces the row-wise MLP embedding encoder with a multi-layer GAT.

    Input:
        mask:              [n_particles, n_genes, n_genes]  soft Gumbel-Softmax samples
        treatment_one_hot: [n_particles, n_genes, n_genes]  identity encoding
        q_mask_logits:     [n_genes, n_genes]               learned edge logits (Ŵ)

    Output:
        mu:    [n_particles, n_genes, n_genes]
        sigma: [n_particles, n_genes, n_genes]

    All particles processed in one batched GAT call (B = n_particles).
    Peak GPU memory: ~30MB for n=655, P=5 (attention map [B,n,n,H]).
    """

    def __init__(
        self,
        n_genes: int,
        n_heads: int = 4,
        n_layers: int = 2,
        d_hidden: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.n_layers = n_layers

        d_in = n_genes * 2  # [mask_row | one_hot]

        self.input_proj = nn.Linear(d_in, d_hidden)

        self.gat_layers = nn.ModuleList([
            GATLayer(
                in_features=d_hidden,
                out_features=d_hidden,
                n_heads=n_heads,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_hidden) for _ in range(n_layers)
        ])

        self.output_head = nn.Linear(d_hidden, n_genes * 2)
        self.var_eps = 1e-4

    def forward(
        self,
        mask: torch.Tensor,              # [P, n_genes, n_genes]
        treatment_one_hot: torch.Tensor, # [P, n_genes, n_genes]
        q_mask_logits: torch.Tensor,     # [n_genes, n_genes]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        P, n_genes, _ = mask.shape

        # v3: Soft adjacency — NO detach: gradient flows back into Ŵ
        # edge_weight = torch.sigmoid(q_mask_logits)  # [n_genes, n_genes]

        # v4 removes soft adjacency
        edge_weight = torch.sigmoid(q_mask_logits).detach() # this is GNN v4

        # Node features: [mask_row | one_hot] → [P, n_genes, n_genes*2]
        node_feat = torch.cat([mask, treatment_one_hot], dim=-1)

        # Input projection: flatten particles for linear, then restore
        H = F.elu(self.input_proj(node_feat.view(P * n_genes, n_genes * 2)))
        H = H.view(P, n_genes, -1)  # [P, n_genes, d_hidden]

        # GAT layers — GATLayer handles B=P natively, no Python loop over particles
        for gat, ln in zip(self.gat_layers, self.layer_norms):
            H_new = F.elu(gat(H, edge_weight))  # [P, n_genes, d_hidden]
            H = ln(H + H_new)                    # residual + layer norm

        # Output head → split into mean and log-variance
        out = self.output_head(H)                # [P, n_genes, n_genes*2]
        mu, log_sigma = out.chunk(2, dim=-1)     # each [P, n_genes, n_genes]
        sigma = F.softplus(log_sigma) + self.var_eps

        return mu, sigma


# ──────────────────────────────────────────────────────────────────────────────
# GNN Guide
# ──────────────────────────────────────────────────────────────────────────────

class gpo_vae_GNNGuide(nn.Module):
    """
    GPO-VAE guide with GNN-based embedding encoder.

    Drop-in replacement for gpo_vae_CorrelatedNormalGuide.
    The only structural change: the row-wise MLP embedding_encoder is replaced
    by a GNNEmbeddingEncoder (multi-layer GAT). Everything else is identical.

    Config usage:
        guide: gpo_vae_GNNGuide
        guide_kwargs:
            ... (same as CorrelatedNormalGuide)
            gnn_n_heads: 4        # optional, default 4
            gnn_n_layers: 2       # optional, default 2
            gnn_d_hidden: 256     # optional, default 256
            gnn_dropout: 0.0      # optional, default 0.0
    """

    def __init__(
        self,
        n_latent: int,
        n_treatments: int,
        n_phenos: int,
        n_qc: int,
        basal_encoder_n_layers: int,
        basal_encoder_n_hidden: int,
        basal_encoder_input_normalization: Optional[Literal["standardize", "log_standardize"]],
        embedding_encoder_n_layers: int,
        embedding_encoder_n_hidden: int,
        x_normalization_stats: Optional[ObservationNormalizationStatistics],
        mask_init: float = 0,
        logits_or_probs: str = "logits",
        gs_temperature: float = 1,
        mean_field_encoder: bool = False,
        knowledge_path: str = None,
        fc_criteria: float = 0,
        # GNN-specific hyperparameters
        gnn_n_heads: int = 4,
        gnn_n_layers: int = 2,
        gnn_d_hidden: int = 256,
        gnn_dropout: float = 0.0,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.n_treatments = n_treatments
        self.n_phenos = n_phenos
        self.n_qc = n_qc
        self.basal_encoder_input_normalization = basal_encoder_input_normalization
        self.x_normalization_stats = x_normalization_stats
        self.mean_field_encoder = mean_field_encoder
        self.param_dict = torch.nn.ParameterDict()
        self.logits_or_probs = logits_or_probs

        # ── q(M) parameters — same as original ───────────────────────────────
        if knowledge_path is None:
            self.param_dict[f"q_mask_{logits_or_probs}"] = torch.nn.Parameter(
                mask_init * torch.ones((n_treatments, n_treatments))
            )
        elif logits_or_probs == "logits":
            self.param_dict[f"q_mask_{logits_or_probs}"] = torch.nn.Parameter(
                probs_to_logits(torch.from_numpy(np.load(knowledge_path))).float()
            )
        elif logits_or_probs == "probs":
            self.param_dict[f"q_mask_{logits_or_probs}"] = torch.nn.Parameter(
                torch.from_numpy(np.load(knowledge_path)).float()
            )

        # ── q(E|M): GNN instead of row-wise MLP ──────────────────────────────
        self.gnn_encoder = GNNEmbeddingEncoder(
            n_genes=n_treatments,
            n_heads=gnn_n_heads,
            n_layers=gnn_n_layers,
            d_hidden=gnn_d_hidden,
            dropout=gnn_dropout,
        )

        self.register_buffer("treatment_one_hot", torch.eye(n_treatments))

        # ── Normalization for basal encoder ───────────────────────────────────
        if self.basal_encoder_input_normalization is None:
            self.normalization_module = None
        else:
            assert x_normalization_stats is not None
            self.normalization_module = get_normalization_module(
                key=self.basal_encoder_input_normalization,
                normalization_stats=x_normalization_stats,
            )

        # ── Artifact encoder (qc_E) — same as original ────────────────────────
        self.register_buffer("qc_encoder_input", torch.ones((n_qc, n_latent)))
        self.qc_encoder = get_likelihood_mlp(
            likelihood_key="normal",
            n_input=n_latent,
            n_output=n_latent,
            n_layers=embedding_encoder_n_layers,
            n_hidden=embedding_encoder_n_hidden,
            use_batch_norm=False,
        )

        # ── Basal state encoder q(z_basal | X, M, E) — same as original ──────
        self.z_basal_encoder = get_likelihood_mlp(
            likelihood_key="normal",
            n_input=n_phenos if mean_field_encoder else n_phenos + n_treatments + n_latent,
            n_output=n_latent,
            n_layers=basal_encoder_n_layers,
            n_hidden=basal_encoder_n_hidden,
            use_batch_norm=False,
        )

        self.register_buffer("gs_temperature", gs_temperature * torch.ones((1,)))
        self.var_eps = 1e-4

    def get_var_keys(self):
        return ["z_basal", "E", "mask", "qc_E"]

    def forward(
        self,
        X: Optional[torch.Tensor] = None,
        D: Optional[torch.Tensor] = None,
        qc: Optional[torch.Tensor] = None,
        condition_values: Optional[Dict[str, torch.Tensor]] = None,
        n_particles: int = 1,
    ) -> Tuple[Dict[str, torch.distributions.Distribution], Dict[str, torch.Tensor]]:

        if condition_values is None:
            condition_values = dict()

        guide_distributions: Dict[str, torch.distributions.Distribution] = {}
        guide_samples: Dict[str, torch.Tensor] = {}

        # ── q(M): Gumbel-Softmax Bernoulli — identical to original ───────────
        q_mask = self.sanitize_tensor(self.param_dict[f"q_mask_{self.logits_or_probs}"])
        if self.logits_or_probs == "logits":
            guide_distributions["q_mask"] = GumbelSoftmaxBernoulliStraightThrough(
                temperature=self.gs_temperature,
                logits=q_mask,
            )
        elif self.logits_or_probs == "probs":
            guide_distributions["q_mask"] = GumbelSoftmaxBernoulliStraightThrough(
                temperature=self.gs_temperature,
                probs=q_mask.clamp(max=1, min=0),
            )

        if "mask" not in condition_values:
            guide_samples["mask"] = guide_distributions["q_mask"].rsample((n_particles,))
        else:
            guide_samples["mask"] = condition_values["mask"]

        # ── q(E|M): GNN — all particles batched in one forward call ──────────
        treatment_one_hot = self.treatment_one_hot.unsqueeze(0).expand(n_particles, -1, -1)

        mu_E, sigma_E = self.gnn_encoder(
            mask=guide_samples["mask"],
            treatment_one_hot=treatment_one_hot,
            q_mask_logits=self.param_dict[f"q_mask_{self.logits_or_probs}"],
        )

        guide_distributions["q_E"] = Normal(mu_E, sigma_E)

        if "E" not in condition_values:
            guide_samples["E"] = guide_distributions["q_E"].rsample()
        else:
            guide_samples["E"] = condition_values["E"]

        # ── Expose H_L for the GRN loss ───────────────────────────────────────
        # mu_E: [n_particles, n_genes, n_genes]
        # loss_modules.py does H_L.mean(0) → [n_genes, n_genes] then D @ H_L
        guide_distributions["H_L"] = mu_E

        # ── q(qc_E): artifact encoder — identical to original ─────────────────
        qc_encoder_input = self.qc_encoder_input.unsqueeze(0).expand(n_particles, -1, -1)
        guide_distributions["q_qc_E"] = self.qc_encoder(qc_encoder_input)
        if "qc_E" not in condition_values:
            guide_samples["qc_E"] = guide_distributions["q_qc_E"].rsample()
        else:
            guide_samples["qc_E"] = condition_values["qc_E"]

        # ── q(z_basal | X, M, E): basal encoder — identical to original ───────
        if X is not None and D is not None and qc is not None:
            encoder_input = X
            if self.normalization_module is not None:
                encoder_input = self.normalization_module(encoder_input)
            encoder_input = torch.unsqueeze(encoder_input, dim=0).expand(n_particles, -1, -1)

            if not self.mean_field_encoder:
                latent_offset = torch.matmul(D, guide_samples["mask"] * guide_samples["E"])
                latent_qc_offset = torch.matmul(qc, guide_samples["qc_E"])
                latent_qc_cf_offset = torch.matmul((1 - qc), guide_samples["qc_E"])

                encoder_cf_input = torch.cat([encoder_input, latent_offset, latent_qc_cf_offset], dim=-1)
                guide_distributions["q_z_cf_basal"] = self.z_basal_encoder(encoder_cf_input)
                encoder_input = torch.cat([encoder_input, latent_offset, latent_qc_offset], dim=-1)

            guide_distributions["q_z_basal"] = self.z_basal_encoder(encoder_input)
            guide_samples["z_basal"] = guide_distributions["q_z_basal"].rsample()

        if "z_basal" in condition_values:
            guide_samples["z_basal"] = condition_values["z_basal"]

        return guide_distributions, guide_samples

    def sanitize_tensor(self, tensor):
        tensor = torch.where(torch.isnan(tensor), torch.full_like(tensor, 0.0), tensor)
        tensor = torch.where(torch.isinf(tensor), torch.full_like(tensor, 0.0), tensor)
        return tensor