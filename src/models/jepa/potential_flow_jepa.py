"""Potential-Flow Joint Embedding Predictive Architecture (PF-JEPA).

Replaces unconstrained neural velocity fields with a Conservative-Dissipative
Scalar Potential Field:
    v_psi(z_t, t, z_ctx) = -nabla_{z_t} Phi_psi(z_t, t, z_ctx, p_regime)

Key Mathematical Invariants:
1. Conservative-Dissipative Flow: Vector field is exact gradient of scalar energy Phi_psi.
2. Exact Energy Curvature: The velocity Jacobian trace equals the scalar Laplacian
   Tr(nabla_z v_psi) = -Delta_{z_t} Phi_psi, computed via Hutchinson trace estimation.
3. Harmonic Grassmannian Regime Conditioning: Discrete modes are represented by
   orthonormal subspace frames U_k in Gr(d, D) regularized via Delsarte frame separation.
4. Dual-Geometry Anomaly Scoring: Combines geodesic midpoint velocity deviation D_M
   with local manifold energy curvature Delta Phi_psi.
"""

from __future__ import annotations

import math
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._jepa_utils import JEPABase, fit_covariance_batched
from src.models.geometric_layers import MovingTangentProjection
from src.models.jepa.flow_ts_jepa import TimestepEmbedding, flow_matching_vicreg_loss, von_neumann_operator_entropy_loss


class HarmonicGrassmannianCodebook(nn.Module):
    """Harmonic Grassmannian Subspace Codebook for Discrete Operational Regimes.
    
    Maintains K orthonormal subspace frames U_k in Gr(d, D) (D x d matrices with U_k^T U_k = I_d).
    Selects active regime via maximum subspace energy capture:
        r* = argmax_k || U_k^T z_ctx ||_2^2
    Regularized with Delsarte spherical frame orthogonality barrier:
        L_grassmann = sum_{i != j} || U_i^T U_j ||_F^2
    """

    def __init__(
        self,
        latent_dim: int,
        n_regimes: int = 4,
        subspace_dim: int = 8,
        temperature: float = 0.1,
        load_balance_weight: float = 0.0,
        hard_eval: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_regimes = n_regimes
        self.subspace_dim = min(subspace_dim, latent_dim)
        self.temperature = temperature
        self.load_balance_weight = load_balance_weight
        self.hard_eval = hard_eval

        raw_frames = torch.randn(n_regimes, latent_dim, self.subspace_dim)
        q_frames = []
        for k in range(n_regimes):
            q, _ = torch.linalg.qr(raw_frames[k])
            q_frames.append(q[:, :self.subspace_dim])
        self.raw_frames = nn.Parameter(torch.stack(q_frames, dim=0))

    @torch.no_grad()
    def init_spectral_band_frames(self, z_samples: torch.Tensor) -> None:
        """Initialize regime frames from interleaved principal-component bands.

        Assigns each regime K a distinct strided slice {k, k+K, k+2K, ...} of
        the descending-variance eigenbasis of the context covariance, so every
        frame starts with comparable captured energy on real data instead of
        QR-random orientations (which leaves the argmax margin noise-dominated).
        """
        Z = z_samples.detach().float()
        Zc = Z - Z.mean(dim=0, keepdim=True)
        cov = (Zc.T @ Zc) / max(Z.size(0) - 1, 1)
        _, evecs = torch.linalg.eigh(cov)
        pcs = evecs.flip(-1)  # columns sorted by descending eigenvalue
        D = pcs.size(0)

        frames = torch.empty(self.n_regimes, D, self.subspace_dim, device=Z.device)
        for k in range(self.n_regimes):
            cols = list(range(k, min(k + self.subspace_dim * self.n_regimes, D), self.n_regimes))
            band = pcs[:, cols]
            if band.size(1) < self.subspace_dim:
                pad = torch.randn(D, self.subspace_dim - band.size(1), device=Z.device)
                band = torch.cat([band, pad], dim=1)
            q, _ = torch.linalg.qr(band)
            frames[k] = q[:, :self.subspace_dim]
        self.raw_frames.copy_(frames.to(device=self.raw_frames.device, dtype=self.raw_frames.dtype))

    def get_orthonormal_frames(self) -> torch.Tensor:
        """Compute orthonormal bases for each regime via QR decomposition.
        
        Returns:
            Tensor of shape (K, D, d) where each slice U_k satisfies U_k^T U_k = I_d.
        """
        frames = []
        for k in range(self.n_regimes):
            q, _ = torch.linalg.qr(self.raw_frames[k])
            frames.append(q[:, :self.subspace_dim])
        return torch.stack(frames, dim=0)

    def forward(
        self,
        z_ctx: torch.Tensor,
        hard: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project context into harmonic regime subspaces.
        
        Args:
            z_ctx: Context representations, shape (B, D)
            hard: Whether to use hard argmax routing during inference
            
        Returns:
            p_ctx: Regime-projected context representations, shape (B, D)
            regime_probs: Probability distribution over regimes, shape (B, K)
            loss_ortho: Delsarte frame orthogonality regularization loss, plus
                the load-balancing auxiliary term when load_balance_weight > 0
        """
        B, D = z_ctx.shape
        frames = self.get_orthonormal_frames()  # (K, D, d)

        energies = []
        projections = []
        for k in range(self.n_regimes):
            U_k = frames[k]  # (D, d)
            proj = z_ctx @ U_k  # (B, d)
            energy = torch.sum(proj ** 2, dim=-1)  # (B,)
            energies.append(energy)
            reconstructed = proj @ U_k.T  # (B, D)
            projections.append(reconstructed)

        energy_tensor = torch.stack(energies, dim=-1)  # (B, K)
        proj_tensor = torch.stack(projections, dim=1)  # (B, K, D)

        if hard or (not self.training and self.hard_eval):
            regime_idx = torch.argmax(energy_tensor, dim=-1)  # (B,)
            regime_probs = F.one_hot(regime_idx, num_classes=self.n_regimes).float()
            p_ctx = proj_tensor[torch.arange(B, device=z_ctx.device), regime_idx]
        else:
            regime_probs = F.softmax(energy_tensor / self.temperature, dim=-1)  # (B, K)
            p_ctx = torch.sum(regime_probs.unsqueeze(-1) * proj_tensor, dim=1)  # (B, D)

        loss_ortho = torch.tensor(0.0, device=z_ctx.device)
        if self.n_regimes > 1:
            for i in range(self.n_regimes):
                for j in range(i + 1, self.n_regimes):
                    overlap = frames[i].T @ frames[j]  # (d, d)
                    loss_ortho = loss_ortho + torch.sum(overlap ** 2)
            loss_ortho = loss_ortho / (self.n_regimes * (self.n_regimes - 1) / 2.0)

        # Switch-style load-balancing auxiliary loss: K * sum_k f_k * P_k,
        # minimized (=1.0) when hard usage f_k and mean soft mass P_k are uniform.
        # f_k is detached (argmax is non-differentiable); gradients flow through
        # P_k to raise under-used frames' energies and deflate over-used ones.
        if self.load_balance_weight > 0 and self.training and not hard:
            f_k = F.one_hot(
                torch.argmax(energy_tensor, dim=-1), num_classes=self.n_regimes
            ).float().mean(dim=0)
            P_k = regime_probs.mean(dim=0)
            loss_lb = self.n_regimes * torch.sum(f_k.detach() * P_k)
            loss_ortho = loss_ortho + self.load_balance_weight * loss_lb

        return p_ctx, regime_probs, loss_ortho


class ScalarPotentialField(nn.Module):
    """Scalar Energy Potential Field Phi_psi(z_t, t, z_ctx, p_regime): R^D -> R.
    
    Uses smooth SiLU activations to ensure continuous first and second
    derivatives for exact Hamiltonian-dissipative velocity and Laplacian computation.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_embed = TimestepEmbedding(embed_dim=hidden_dim)

        self.input_proj = nn.Linear(latent_dim * 3, hidden_dim)

        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "norm": nn.LayerNorm(hidden_dim),
                        "linear1": nn.Linear(hidden_dim, hidden_dim),
                        "act": nn.SiLU(),
                        "linear2": nn.Linear(hidden_dim, hidden_dim),
                        "dropout": nn.Dropout(dropout),
                        "time_proj": nn.Linear(hidden_dim, hidden_dim),
                    }
                )
            )

        self.out_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.out_scalar = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        z_ctx: torch.Tensor,
        p_regime: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute scalar potential energy: (B, D) x (B,) x (B, D) x (B, D) -> (B, 1)."""
        if t.ndim == 0 or (t.ndim == 1 and t.size(0) == 1 and z_t.size(0) > 1):
            t = t.expand(z_t.size(0))

        if p_regime is None:
            p_regime = torch.zeros_like(z_ctx)

        t_emb = self.time_embed(t)
        x_in = torch.cat([z_t, z_ctx, p_regime], dim=-1)
        h = self.input_proj(x_in)

        for block in self.blocks:
            res = h
            out = block["norm"](h)
            out = out + block["time_proj"](t_emb)
            out = out + block["act"](block["linear1"](out))
            out = block["dropout"](block["linear2"](out))
            h = res + out

        h = self.out_norm(h)
        phi = self.out_scalar(h)
        return phi


class PotentialFlowPredictor(nn.Module):
    """Continuous Velocity Predictor derived from Scalar Potential Field.
    
    Velocity vector is the negative gradient of the scalar potential:
        v_psi(z_t, t, z_ctx, p_regime) = -nabla_{z_t} Phi_psi
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.potential_field = ScalarPotentialField(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.tangent_proj = MovingTangentProjection(latent_dim=latent_dim)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        z_ctx: torch.Tensor,
        p_regime: Optional[torch.Tensor] = None,
        create_graph: bool = True,
    ) -> torch.Tensor:
        """Compute conservative-dissipative velocity vector v_t = -nabla_{z_t} Phi_psi
        projected onto the tangent bundle of S^{D-1} (Chapter 2, §4–§5).
        """
        with torch.enable_grad():
            z_t_in = z_t if z_t.requires_grad else z_t.clone().detach().requires_grad_(True)
            phi = self.potential_field(z_t_in, t, z_ctx, p_regime)
            grad = torch.autograd.grad(
                outputs=phi.sum(),
                inputs=z_t_in,
                create_graph=create_graph,
                retain_graph=True if create_graph else False,
                only_inputs=True,
            )[0]
        v_raw = -grad
        # Project onto tangent space T_{z_t} S^{D-1}
        return self.tangent_proj(v_raw, z_pole=z_t)


    def compute_energy_laplacian(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        z_ctx: torch.Tensor,
        p_regime: Optional[torch.Tensor] = None,
        n_probes: int = 1,
    ) -> torch.Tensor:
        """Compute Energy Laplacian Curvature Delta_{z_t} Phi_psi = Tr(nabla_z^2 Phi_psi)."""
        with torch.enable_grad():
            B, D = z_t.shape
            z_t_in = z_t.clone().detach().requires_grad_(True)
            phi = self.potential_field(z_t_in, t, z_ctx, p_regime)
            grad_phi = torch.autograd.grad(
                outputs=phi.sum(),
                inputs=z_t_in,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            
            total_trace = torch.zeros(B, device=z_t.device, dtype=z_t.dtype)
            for _ in range(n_probes):
                u = torch.randn_like(z_t_in)
                inner = torch.sum(grad_phi * u)
                jvp = torch.autograd.grad(
                    outputs=inner,
                    inputs=z_t_in,
                    create_graph=False,
                    retain_graph=True,
                    only_inputs=True,
                )[0]
                quad = torch.sum(jvp * u, dim=-1)
                total_trace = total_trace + quad
                
            return total_trace / float(n_probes)



class PotentialFlowJEPAModel(JEPABase):
    """Potential-Flow Joint Embedding Predictive Architecture (PF-JEPA)."""

    def __init__(
        self,
        context_encoder: nn.Module,
        latent_dim: int,
        predictor_hidden_dim: int = 64,
        predictor_layers: int = 3,
        n_regimes: int = 4,
        subspace_dim: int = 8,
        use_regimes: bool = True,
        ema_decay: float = 0.995,
        dropout: float = 0.05,
        codebook_temperature: float = 0.1,
        load_balance_weight: float = 0.0,
        eval_hard_routing: bool = True,
    ):
        super().__init__()
        self.context_encoder = context_encoder
        self.latent_dim = latent_dim
        self.use_regimes = use_regimes
        self.ema_decay = ema_decay
        self.eval_hard_routing = eval_hard_routing

        self.target_encoder = self.init_target_encoder(context_encoder)

        if use_regimes and n_regimes > 1:
            self.grassmannian_codebook = HarmonicGrassmannianCodebook(
                latent_dim=latent_dim,
                n_regimes=n_regimes,
                subspace_dim=subspace_dim,
                temperature=codebook_temperature,
                load_balance_weight=load_balance_weight,
                hard_eval=eval_hard_routing,
            )
        else:
            self.grassmannian_codebook = None

        self.flow_predictor = PotentialFlowPredictor(
            latent_dim=latent_dim,
            hidden_dim=predictor_hidden_dim,
            num_layers=predictor_layers,
            dropout=dropout,
        )

        from src.models.geometric_layers import CoordinateSaliencyGate
        self.saliency_gate = CoordinateSaliencyGate(dim=latent_dim, tau=0.5, alpha=0.5)

        self.register_mahalanobis_buffers(latent_dim)

    def forward(
        self,
        context_windows: torch.Tensor,
        target_windows: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        z_noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        B = context_windows.size(0)
        device = context_windows.device

        z_ctx = self.context_encoder(context_windows)

        loss_grassmann = torch.tensor(0.0, device=device)
        if self.grassmannian_codebook is not None:
            p_regime, _, loss_grassmann = self.grassmannian_codebook(z_ctx)
        else:
            p_regime = torch.zeros_like(z_ctx)

        if target_windows is None:
            return z_ctx, None, None, None, loss_grassmann

        self.target_encoder.eval()
        with torch.no_grad():
            z_tgt_true = self.target_encoder(target_windows)

        if t is None:
            t = torch.rand(B, device=device, dtype=context_windows.dtype)

        if z_noise is None:
            z_noise = torch.randn(B, self.latent_dim, device=device, dtype=context_windows.dtype)

        t_expand = t.view(B, 1)
        z_t = (1.0 - t_expand) * z_noise + t_expand * z_tgt_true
        v_target = z_tgt_true - z_noise

        v_pred = self.flow_predictor(z_t, t, z_ctx, p_regime=p_regime, create_graph=self.training)

        return z_ctx, z_tgt_true, v_pred, v_target, loss_grassmann

    def compute_objective(
        self,
        ctx: torch.Tensor,
        tgt: torch.Tensor,
        config=None,
        grassmann_weight: float = 0.0,
        cov_weight: float = 0.0,
        var_weight: float = 0.0,
        flow_weight: float = 1.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        if not self.training:
            B = ctx.size(0)
            device = ctx.device
            dtype = ctx.dtype
            t_val = torch.full((B,), 0.5, device=device, dtype=dtype)
            z_zero = torch.zeros(B, self.latent_dim, device=device, dtype=dtype)
            z_ctx, z_tgt_true, v_pred, v_target, loss_grass = self.forward(ctx, tgt, t=t_val, z_noise=z_zero)
        else:
            z_ctx, z_tgt_true, v_pred, v_target, loss_grass = self.forward(ctx, tgt)

        # 1. Pure Optimal Transport velocity matching on the tangent bundle S^{D-1}
        effective_flow_weight = flow_weight if config is None else getattr(config, "vicreg_sim_weight", flow_weight)
        loss_flow = F.mse_loss(v_pred, v_target)
        total_loss = effective_flow_weight * loss_flow

        loss_metrics = {
            "total_loss": float(total_loss.item()),
            "loss_flow": float(loss_flow.item()),
            "loss_var": 0.0,
            "loss_cov": 0.0,
            "loss_grassmann": float(loss_grass.item()) if isinstance(loss_grass, torch.Tensor) else 0.0,
        }

        # Optional backward-compatible regularization if explicitly requested by config/caller
        effective_var = var_weight if config is None else getattr(config, "vicreg_var_weight", var_weight)
        effective_cov = cov_weight if config is None else getattr(config, "vicreg_cov_weight", cov_weight)
        if effective_var > 0 or effective_cov > 0:
            reg_loss, aux_metrics = flow_matching_vicreg_loss(
                v_pred=v_pred,
                v_target=v_target,
                z_ctx=z_ctx,
                z_tgt_true=z_tgt_true,
                flow_weight=effective_flow_weight,
                var_weight=effective_var,
                cov_weight=effective_cov,
            )
            total_loss = reg_loss
            loss_metrics.update(aux_metrics)

        if self.grassmannian_codebook is not None and grassmann_weight > 0:
            total_loss = total_loss + grassmann_weight * loss_grass
            loss_metrics["loss_grassmann"] = float(loss_grass.item())
            loss_metrics["total_loss"] = float(total_loss.item())

        return total_loss, loss_metrics

    @torch.no_grad()
    def compute_predictive_discrepancy(
        self,
        context_windows: torch.Tensor,
        observed_target_windows: torch.Tensor,
        use_mahalanobis: bool = False,
        include_curvature: bool = False,
        curvature_weight: float = 0.10,
        integration_steps: int = 1,
    ) -> torch.Tensor:
        if use_mahalanobis and not bool(self.precision_fitted.item()):
            raise RuntimeError("Mahalanobis scoring requested, but covariance has not been fitted.")

        self.eval()
        B = context_windows.size(0)
        device = context_windows.device
        dtype = context_windows.dtype

        z_ctx = self.context_encoder(context_windows)
        z_tgt = self.target_encoder(observed_target_windows)

        if self.grassmannian_codebook is not None:
            p_regime, _, _ = self.grassmannian_codebook(z_ctx, hard=self.eval_hard_routing)
        else:
            p_regime = torch.zeros_like(z_ctx)

        if integration_steps > 1:
            dt = 1.0 / integration_steps
            v_acc = torch.zeros_like(z_tgt)
            for step in range(integration_steps):
                t_k = torch.full((B,), (step + 0.5) * dt, device=device, dtype=dtype)
                z_k = (step + 0.5) * dt * z_tgt
                v_k = self.flow_predictor(z_k, t_k, z_ctx, p_regime=p_regime, create_graph=False)
                v_acc = v_acc + v_k
            v_pred = v_acc / integration_steps
            diff = v_pred - z_tgt
            t_eval = torch.full((B,), 0.5, device=device, dtype=dtype)
            z_eval = 0.5 * z_tgt
        else:
            t_eval = torch.full((B,), 0.5, device=device, dtype=dtype)
            z_eval = 0.5 * z_tgt
            v_pred = self.flow_predictor(z_eval, t_eval, z_ctx, p_regime=p_regime, create_graph=False)
            diff = v_pred - z_tgt

        if use_mahalanobis and bool(self.precision_fitted.item()):
            diff_c = diff - self.residual_mean
            mahal_coord = (diff_c @ self.precision_matrix) * diff_c
            e_white = torch.sign(diff_c) * torch.sqrt(torch.clamp(mahal_coord, min=0.0))
            base_score = self.saliency_gate(e_white)
        else:
            base_score = self.saliency_gate(diff)

        if include_curvature:
            with torch.enable_grad():
                laplacian = self.flow_predictor.compute_energy_laplacian(
                    z_eval, t_eval, z_ctx, p_regime=p_regime, n_probes=1
                )
            curv_score = F.relu(laplacian)
            total_score = base_score + curvature_weight * curv_score
        else:
            total_score = base_score

        return total_score

    @torch.no_grad()
    def fit_mahalanobis_covariance(
        self,
        context_windows,
        target_windows,
        batch_size: int = 512,
        reg: float = 1e-3,
        integration_steps: int = 1,
    ) -> None:
        def residual_fn(ctx_b, tgt_b):
            z_ctx = self.context_encoder(ctx_b)
            z_tgt = self.target_encoder(tgt_b)
            B_b = ctx_b.size(0)
            if self.grassmannian_codebook is not None:
                p_regime, _, _ = self.grassmannian_codebook(z_ctx, hard=self.eval_hard_routing)
            else:
                p_regime = torch.zeros_like(z_ctx)
            if integration_steps > 1:
                dt = 1.0 / integration_steps
                v_acc = torch.zeros_like(z_tgt)
                for step in range(integration_steps):
                    t_k = torch.full((B_b,), (step + 0.5) * dt, device=ctx_b.device, dtype=ctx_b.dtype)
                    z_k = (step + 0.5) * dt * z_tgt
                    v_k = self.flow_predictor(z_k, t_k, z_ctx, p_regime=p_regime, create_graph=False)
                    v_acc = v_acc + v_k
                v_pred = v_acc / integration_steps
                return v_pred - z_tgt
            else:
                t_mid = torch.full((B_b,), 0.5, device=ctx_b.device, dtype=ctx_b.dtype)
                z_mid = 0.5 * z_tgt
                v_pred = self.flow_predictor(z_mid, t_mid, z_ctx, p_regime=p_regime, create_graph=False)
                return v_pred - z_tgt


        fit_covariance_batched(
            self,
            context_windows,
            target_windows,
            residual_fn=residual_fn,
            dim=self.latent_dim,
            batch_size=batch_size,
            reg=reg,
            precision_buffer=self.precision_matrix,
            residual_mean_buffer=self.residual_mean,
            fitted_buffer=self.precision_fitted,
        )


PotentialFlowJEPA = PotentialFlowJEPAModel
PF_JEPA = PotentialFlowJEPAModel
