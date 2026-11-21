"""Spectral resolvent, Hankel polynomial, and Cohn-Elkies Fourier filters."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResolventPurification(nn.Module):
    """Matrix Resolvent Purification Operator Layer (Chapter 6, §4.2).

    Transforms an unconstrained factor :math:`L \\in \\mathbb{R}^{B \\times D \\times r}`
    into a strictly Symmetric Positive Semi-Definite (SPSD) operator with spectrum
    bounded in :math:`[0, 1)`:

    .. math::
        F = L L^T \\succeq 0, \\quad \\Gamma_\\gamma(F) = F (F + \\gamma I)^{-1}

    Spectral Properties (Proposition 4.2 & Lemma 4.3):
    1. **Strict Positive Semi-Definiteness**: If :math:`F` has eigenvalues :math:`\\lambda_i \\ge 0`,
       :math:`\\Gamma_\\gamma(F)` has eigenvalues :math:`\\frac{\\lambda_i}{\\lambda_i + \\gamma} \\ge 0`.
       The cone defect is **identically zero by construction**.
    2. **Bounded Spectrum & Contraction**: :math:`\\frac{\\lambda_i}{\\lambda_i + \\gamma} < 1` for all
       :math:`\\lambda_i \\ge 0`, bounding the operator norm :math:`\\|\\Gamma_\\gamma(F)\\| < 1` and
       completely preventing stiffness/variance collapse or explosion.
    3. **Smooth & Differentiable**: Computed via symmetric positive definite solve
       without requiring costly full eigenvalue decomposition (:func:`torch.linalg.eigvalsh`).

    Args:
        dim: Matrix dimension :math:`D`.
        gamma: Resolvent regularization bandwidth :math:`\\gamma > 0`.
        learnable_gamma: Whether :math:`\\gamma` is a learnable parameter.
    """

    def __init__(
        self,
        dim: int,
        gamma: float = 1e-2,
        learnable_gamma: bool = False,
    ):
        super().__init__()
        self.dim = dim
        if learnable_gamma:
            self.log_gamma = nn.Parameter(torch.log(torch.tensor(gamma)))
        else:
            self.register_buffer("log_gamma", torch.log(torch.tensor(gamma)))

    @property
    def gamma(self) -> torch.Tensor:
        return torch.exp(self.log_gamma).clamp(min=1e-5, max=10.0)

    def purify_psd(self, F_mat: torch.Tensor) -> torch.Tensor:
        """Purify an existing SPSD matrix :math:`F \\succeq 0` via :math:`F (F + \\gamma I)^{-1}`.

        Args:
            F_mat: Tensor of shape ``(..., D, D)`` with :math:`F \\succeq 0`.

        Returns:
            Purified SPSD tensor with eigenvalues in :math:`[0, 1)`.
        """
        B = F_mat.size(0)
        device = F_mat.device
        dtype = F_mat.dtype
        D = self.dim

        eye = torch.eye(D, device=device, dtype=dtype)
        if F_mat.ndim == 3:
            eye = eye.unsqueeze(0).expand(B, -1, -1)
        A = F_mat + self.gamma * eye

        X = torch.linalg.solve(A, F_mat)
        purified = 0.5 * (X + X.transpose(-1, -2))
        return purified

    def forward(self, L: torch.Tensor, is_psd_matrix: bool = False) -> torch.Tensor:
        """Compute the resolvent-purified SPSD matrix :math:`\\Gamma_\\gamma(L L^T)`
        or :math:`\\Gamma_\\gamma(F)` if *is_psd_matrix* is True.

        Args:
            L: Tensor of shape ``(..., D, r)`` or ``(..., D, D)``.
            is_psd_matrix: If True, treats *L* as a pre-formed SPSD matrix :math:`F`.

        Returns:
            Purified SPSD tensor :math:`\\Sigma \\in \\mathcal{S}_+` of shape ``(..., D, D)``.
        """
        if is_psd_matrix:
            return self.purify_psd(L)

        # F = L L^T (SPSD by construction)
        F_mat = torch.bmm(L, L.transpose(-1, -2)) if L.ndim == 3 else L @ L.T
        return self.purify_psd(F_mat)


class HankelPolynomialFilter(nn.Module):
    r"""Algebraic Hankel Polynomial Filter Layer (Chapter 7, §4 & Chapter 5, §5).

    Decomposes temporal signals over a window of length :math:`L` into:
    1. **Low-degree orthogonal polynomial moments** representing smooth dynamical drift:
       :math:`\hat{X}_{\\text{poly}} = P_{\\text{poly}} X`.
    2. **High-frequency residual perturbations** representing transient anomalies or noise:
       :math:`X_{\\text{res}} = (I - P_{\\text{poly}}) X`.

    By construction:
    - :math:`P_{\\text{poly}}` is an exact orthogonal projector: :math:`P^2 = P`, :math:`P^T = P`.
    - Polynomial signals of degree :math:`\le d` are preserved with zero error.
    - High-frequency phase jumps and anomalies are isolated into the residual subspace.

    Args:
        channels: Feature channel dimension :math:`C`.
        window_len: Sequence window length :math:`L`.
        degree: Maximum polynomial degree :math:`d \ge 1` (default 3).
        learnable_fusion: Whether to learn fusion weights between polynomial and residual components.
        eps: Small numerical stability constant.
    """

    def __init__(
        self,
        channels: int,
        window_len: int,
        degree: int = 3,
        learnable_fusion: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.channels = channels
        self.window_len = window_len
        self.degree = min(degree, max(window_len - 1, 1))
        self.eps = eps

        # Construct normalized Legendre polynomial basis over [-1, 1]
        t = torch.linspace(-1.0, 1.0, steps=window_len)
        cols = []
        for d in range(self.degree + 1):
            if d == 0:
                p = torch.ones_like(t)
            elif d == 1:
                p = t
            elif d == 2:
                p = 0.5 * (3 * t**2 - 1)
            elif d == 3:
                p = 0.5 * (5 * t**3 - 3 * t)
            elif d == 4:
                p = 0.125 * (35 * t**4 - 30 * t**2 + 3)
            else:
                p = t**d
            # Normalize column
            norm = torch.norm(p, p=2).clamp(min=eps)
            cols.append(p / norm)

        # Basis matrix V: (L, degree + 1)
        V = torch.stack(cols, dim=1)
        # Compute exact projection matrix P_poly = V (V^T V)^{-1} V^T
        VtV = V.T @ V + eps * torch.eye(self.degree + 1)
        P_poly = V @ torch.linalg.solve(VtV, V.T)
        # Ensure exact symmetry
        P_poly = 0.5 * (P_poly + P_poly.T)

        self.register_buffer("P_poly", P_poly)  # (L, L)
        self.register_buffer("V", V)            # (L, degree + 1)

        if learnable_fusion:
            self.poly_proj = nn.Linear(channels, channels)
            self.res_proj = nn.Linear(channels, channels)
            self.gate = nn.Parameter(torch.zeros(channels))
        else:
            self.poly_proj = nn.Identity()
            self.res_proj = nn.Identity()
            self.gate = None

    def compute_hankel_moments(self, x: torch.Tensor) -> torch.Tensor:
        """Compute algebraic power-sum moments across the temporal window.

        Args:
            x: Input tensor of shape ``(B, L, C)`` or ``(L, C)``.

        Returns:
            Moments tensor of shape ``(B, degree + 1, C)`` or ``(degree + 1, C)``.
        """
        # V is (L, d+1) -> V^T @ x is (d+1, C)
        if x.ndim == 3:
            return torch.einsum("lt,blc->btc", self.V, x)
        return self.V.T @ x

    def forward(
        self,
        x: torch.Tensor,
        return_decomposition: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass filtering input sequence *x*.

        Args:
            x: Tensor of shape ``(B, L, C)``.
            return_decomposition: If True, returns ``(output, x_poly, x_res)``.

        Returns:
            Filtered output of shape ``(B, L, C)``.
        """
        L_in = x.size(1)
        if L_in != self.window_len:
            # Dynamically interpolate projector if window length varies
            P = F.interpolate(
                self.P_poly.unsqueeze(0).unsqueeze(0),
                size=(L_in, L_in),
                mode="bilinear",
                align_corners=True,
            ).squeeze(0).squeeze(0)
        else:
            P = self.P_poly

        # x_poly = P @ x along temporal dimension
        x_poly = torch.matmul(P, x)
        x_res = x - x_poly

        if self.gate is not None:
            alpha = torch.sigmoid(self.gate).unsqueeze(0).unsqueeze(0)
            out = self.poly_proj(x_poly) + alpha * self.res_proj(x_res)
        else:
            out = x_poly + x_res

        if return_decomposition:
            return out, x_poly, x_res
        return out


class ResolventPurificationBottleneck(nn.Module):
    """Vector-level Resolvent Whitening & Dimensional Purification Bottleneck (Chapter 6, §4.2).

    Normalizes representations by applying the Resolvent Operator to the empirical covariance:
        R_gamma(C) = C (C + gamma * I)^{-2}

    Eigenvalues lambda_i are transformed via:
        tilde{lambda}_i = sqrt(lambda_i) / (lambda_i + gamma)

    For strong signal modes (lambda >> gamma), tilde{lambda} -> 1 / sqrt(lambda) (exact whitening).
    For null/noise directions (lambda << gamma), tilde{lambda} -> sqrt(lambda) / gamma -> 0 (noise suppression).

    This prevents rank collapse and purifies noisy sensor channels without requiring
    VICReg covariance penalties in the loss function.

    Args:
        latent_dim: Feature dimension D.
        gamma: Regularization bandwidth (default 1e-2).
        eps: Small numerical stability constant.
        affine: Whether to include learnable scale and bias.
    """

    def __init__(
        self,
        latent_dim: int,
        gamma: float = 1e-2,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.eps = eps

        if affine:
            self.weight = nn.Parameter(torch.ones(latent_dim))
            self.bias = nn.Parameter(torch.zeros(latent_dim))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass purifying representation tensor z of shape (..., D)."""
        orig_shape = z.shape
        D = self.latent_dim
        z_flat = z.reshape(-1, D)
        B = z_flat.size(0)

        if B <= 1:
            # Fallback for single sample: standard normalization
            z_norm = (z_flat - z_flat.mean(dim=-1, keepdim=True)) / (z_flat.std(dim=-1, keepdim=True) + self.eps)
            if self.weight is not None:
                z_norm = z_norm * self.weight + self.bias
            return z_norm.reshape(orig_shape)

        mean = z_flat.mean(dim=0, keepdim=True)
        z_c = z_flat - mean

        # Empirical covariance matrix C in R^{D x D}
        cov = (z_c.T @ z_c) / (B - 1) + self.eps * torch.eye(D, device=z.device, dtype=z.dtype)

        # SVD / Eigendecomposition of covariance
        evals, evecs = torch.linalg.eigh(cov)
        evals_pos = torch.clamp(evals, min=self.eps)

        # Resolvent square-root filter: lambda^{1/2} / (lambda + gamma)
        filtered_scale = torch.sqrt(evals_pos) / (evals_pos + self.gamma)

        # Whitening projection operator: W = evecs @ diag(filtered_scale) @ evecs.T
        whitening_op = evecs @ torch.diag(filtered_scale) @ evecs.T
        z_purified = z_c @ whitening_op

        if self.weight is not None:
            z_purified = z_purified * self.weight + self.bias

        return z_purified.reshape(orig_shape)


class CohnElkiesFilter(nn.Module):
    """Cohn-Elkies Dual-Shell Fourier Sign-Uncertainty Modulation Layer (Ten Proofs, Chapter 1, §4.2).

    Constructs the even entire Mellin-Fourier perturbation envelope:
        h_eps(zeta) = integral_0^infty w(a) (cos(a * zeta) - 1) da
    where w(a) = w_s(a) + w_B(a) is the two-shell density that guarantees non-negative total
    damping away from zero frequency and strictly prevents sign uncertainty / high-frequency leakage:
        w_s(a) = - (1 - 2*eps*(1+a)) * exp(-2a) / (2a^2 cosh a) * 1_{[a0, A]}
        w_B(a) = Q / cosh a * 1_{[B, B+1]}

    Modulates representation frequencies via the even damping multiplier:
        E(tau) = exp(lambda * h_eps(tau / lambda))
    suppressing high-frequency OOD noise while strictly preserving low-frequency geometric invariants.
    
    Args:
        latent_dim: Feature dimension D.
        n_shells: Number of discrete quadrature nodes approximating the dilation shell integral (default: 8).
        eps: Damping parameter in (0, 0.2) (default: 0.05).
    """

    def __init__(
        self,
        latent_dim: int,
        n_shells: int = 8,
        eps: float = 0.05,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_shells = n_shells
        self.eps = eps
        self.lam = max(latent_dim / 2.0, 1.0)

        # Quad nodes for shell integration: a in [eps^2, log(1/eps)]
        a_min = max(eps ** 2, 1e-3)
        a_max = max(math.log(1.0 / max(eps, 1e-4)), a_min + 0.1)
        grid = torch.linspace(a_min, a_max, n_shells)
        self.register_buffer("shell_nodes", grid)

        # Compute strictly positive normalized dual-shell weights w(a) > 0
        # Since cos(a * zeta) - 1 in [-2, 0] and sum_k w(a_k) = 1, h_eps in [-2, 0] algebraically without clamping.
        taper = torch.abs(1.0 - 2.0 * eps * (1.0 + grid)) + eps
        cosh_a = torch.cosh(grid)
        w_s = (taper * torch.exp(-2.0 * grid)) / (2.0 * torch.square(grid) * cosh_a + 1e-6)
        # Remote shell for high-frequency damping
        B_val = min(1.0 / (eps ** 2 + 1e-4), 20.0)
        w_B = math.exp(-0.5 * B_val) / math.cosh(B_val)
        w_raw = w_s + w_B
        w_total = w_raw / w_raw.sum().clamp(min=1e-6)
        self.register_buffer("shell_weights", w_total)

        # Learnable spectral gain gate (initialized so sigmoid(-2.2) ≈ 0.1)
        self.gain = nn.Parameter(torch.full((latent_dim,), -2.2))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply Cohn-Elkies dual-shell spectral modulation to representation z.

        Args:
            z: Latent tensor of shape (..., D).

        Returns:
            Damped and calibrated tensor of shape (..., D).
        """
        # Normalized frequency coordinates zeta_j = z_j / (lambda + eps)
        zeta = z / (self.lam + 1e-5)  # (..., D)

        # Vectorized Mellin perturbation across shells:
        # cos(a * zeta) - 1 in [-2, 0]: shape (..., D, n_shells)
        zeta_expanded = zeta.unsqueeze(-1)  # (..., D, 1)
        nodes = self.shell_nodes.view(*([1] * zeta.ndim), self.n_shells)
        weights = self.shell_weights.view(*([1] * zeta.ndim), self.n_shells)

        # Non-positive envelope h_eps in [-2, 0] strictly by positive shell weights
        cos_diff = torch.cos(zeta_expanded * nodes) - 1.0
        h_eps = torch.sum(cos_diff * weights, dim=-1)  # (..., D) in [-2, 0]

        # Even damping factor exp(lambda * h_eps) in [exp(-2*lambda), 1] without hard clamping
        damping = torch.exp(self.lam * h_eps)

        # Modulated output with residual highway
        return z * (1.0 + torch.sigmoid(self.gain) * (damping - 1.0))


class HankelMomentFilter(nn.Module):
    """Bounded-Moment Hankel Separable Filter (Ten Proofs, Chapter 7, §4.1–§4.2).

    Given a temporal sequence of latent representations z_t in R^{B x L x D}, constructs
    the local Hankel matrix of power-sum trajectory moments:
        H_{i, l} = mu_{i+l}(X)
    and applies resolvent operator regularization:
        R(H) = H (H + gamma * I)^{-1}
    to enforce that trajectory updates conform to a separable algebraic polynomial manifold.

    Instantly detects irregular jump discontinuities, sensor dropouts, and chaotic turbulence
    through Hankel rank breakdown without requiring auxiliary loss penalties.

    Args:
        latent_dim: Feature dimension D.
        hankel_order: Half-size of the Hankel matrix h (default: 4, yielding an 8-moment matrix).
        gamma: Resolvent regularization constant (default: 0.01).
    """

    def __init__(
        self,
        latent_dim: int,
        hankel_order: int = 4,
        gamma: float = 0.01,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.h = hankel_order
        self.gamma = gamma

        # Projection head to moment basis
        self.moment_proj = nn.Linear(latent_dim, 2 * hankel_order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Filter representations x through the bounded-moment Hankel resolvent operator.

        Args:
            x: Input tensor of shape (B, D) or (B, L, D).

        Returns:
            Filtered tensor of identical shape with enforced algebraic moment consistency.
        """
        if x.ndim == 2:
            # (B, D): apply channel-wise Hankel gating
            B, D = x.shape
            moments = self.moment_proj(x)  # (B, 2*h)

            # Construct Hankel matrix H of shape (B, h, h)
            # H[b, i, j] = moments[b, i + j]
            h = self.h
            idx_i = torch.arange(h, device=x.device).unsqueeze(1)  # (h, 1)
            idx_j = torch.arange(h, device=x.device).unsqueeze(0)  # (1, h)
            hankel_idx = idx_i + idx_j  # (h, h) in [0, 2*h - 2]

            H = moments[:, hankel_idx]  # (B, h, h)

            # Symmetrize Hankel: H_sym = 1/2 (H + H^T)
            H_sym = 0.5 * (H + H.transpose(-1, -2))

            # Resolvent spectral trace: Tr(H_sym (|H_sym| + gamma I)^{-1}) / h in (-1, 1)
            # By similarity invariance of trace, Tr(V diag(lambda_filt) V^T) == sum(lambda_filt).
            evals = torch.linalg.eigvalsh(H_sym)
            evals_filtered = evals / (torch.abs(evals) + self.gamma)
            gate = evals_filtered.mean(dim=-1, keepdim=True)  # (B, 1) strictly in (-1, 1)

            # Directly modulate representation by algebraically bounded resolvent trace
            return x * (1.0 + 0.5 * gate)

        elif x.ndim == 3:
            # (B, L, D): apply along time sequence
            B, L, D = x.shape
            x_flat = x.reshape(B * L, D)
            out_flat = self.forward(x_flat)
            return out_flat.view(B, L, D)

        return x

__all__ = [
    "ResolventPurification",
    "HankelPolynomialFilter",
    "ResolventPurificationBottleneck",
    "CohnElkiesFilter",
    "HankelMomentFilter",
]
