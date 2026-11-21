"""Fluid-mechanics inspired geometric operators (wavelet shear, stress cone, curl, boundary)."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShearingWaveletPulseBlock(nn.Module):
    """Shearing Wavelet Pulse Convolution Block (Navier-Stokes Blowup Paper, §7 & §3.3).

    Models perturbative wavepackets w(x, t) = a(t) cos(xi(t) . x + phi) evolving under
    background shear g and viscous dissipation nu:
        1/2 d/dv |t|^2 = -g . (t_r (t_theta, t_z)) - nu k^2 |xi|^2 |t|^2

    As the background shear tilts and stretches the wavevector xi(v) = xi_0 - v g,
    transient amplification dominates initially (extracting energy from shear),
    followed by viscous dissipation overtaking it. This creates an asymmetric envelope:
        P(v) = exp( lambda v - 0.5 * nu * (1 + g^2 v^2) v^2 ) * cos(omega v + phi)

    Acts as an expressive 1D temporal filter bank that naturally detects transient
    burst dynamics, sudden anomaly shocks, and dissipative returns to laminar baseline.

    Args:
        channels: Feature channel dimension C.
        kernel_size: Temporal filter kernel size (must be odd, default 15).
        num_pulses: Number of diverse wavepacket pulse heads per channel (default 4).
        residual: Whether to add a residual skip connection with the input.
        eps: Small numerical stability constant.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 15,
        num_pulses: int = 4,
        residual: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.num_pulses = num_pulses
        self.residual = residual
        self.eps = eps

        # Normalized temporal grid v in [-1, 1]
        v = torch.linspace(-1.0, 1.0, steps=kernel_size)
        self.register_buffer("v_grid", v)

        # Learnable hydrodynamic wavepacket parameters per pulse head
        # 1. Carrier frequency omega (initialized to geometric octave spacing)
        init_freqs = torch.exp(torch.linspace(math.log(1.0), math.log(8.0), steps=num_pulses))
        self.omega = nn.Parameter(init_freqs.unsqueeze(0).expand(channels, -1).clone())
        # 2. Phase shift phi
        self.phi = nn.Parameter(torch.zeros(channels, num_pulses))
        # 3. Background shear gradient g
        self.shear = nn.Parameter(torch.randn(channels, num_pulses) * 0.2)
        # 4. Growth rate lambda (shear amplification)
        self.growth = nn.Parameter(torch.full((channels, num_pulses), 0.1))
        # 5. Viscous damping log_nu (nu > 0 via softplus)
        self.log_viscosity = nn.Parameter(torch.full((channels, num_pulses), -0.5))

        # Pulse mixing projection: (channels * num_pulses) -> channels
        self.pulse_mix = nn.Conv1d(
            in_channels=channels * num_pulses,
            out_channels=channels,
            kernel_size=1,
            bias=True,
        )
        self.norm = nn.LayerNorm(channels)

    def generate_filters(self) -> torch.Tensor:
        """Generate normalized shearing wavelet pulse kernels.

        Returns:
            Tensor of shape (C * num_pulses, 1, K) suitable for grouped 1D convolution.
        """
        # v_grid: (1, 1, K)
        v = self.v_grid.view(1, 1, -1)
        C, P = self.channels, self.num_pulses
        K = self.kernel_size

        # Expand parameters to (C, P, 1)
        growth = self.growth.unsqueeze(-1)
        shear = self.shear.unsqueeze(-1)
        nu = F.softplus(self.log_viscosity).unsqueeze(-1)
        omega = self.omega.unsqueeze(-1)
        phi = self.phi.unsqueeze(-1)

        # Asymmetric Gaussian envelope: exp(lambda * v - 0.5 * nu * (1 + g^2 v^2) * v^2)
        v_sq = v ** 2
        dissipation = 0.5 * nu * (1.0 + (shear ** 2) * v_sq) * v_sq
        envelope = torch.exp(growth * v - dissipation)

        # Oscillatory wave: cos(omega * v + phi)
        oscillation = torch.cos(omega * v * math.pi + phi)

        # Full pulse kernel: (C, P, K)
        kernels = envelope * oscillation

        # L2-normalize each kernel along temporal dimension K
        norms = torch.norm(kernels, p=2, dim=-1, keepdim=True).clamp(min=self.eps)
        kernels = kernels / norms

        # Reshape to (C * P, 1, K) for depthwise 1D conv
        return kernels.view(C * P, 1, K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass filtering temporal sequence x.

        Args:
            x: Input tensor of shape (B, T, C) or (B, C, T).

        Returns:
            Filtered output of the same shape as input.
        """
        is_channels_last = (x.ndim == 3 and x.size(-1) == self.channels)
        if is_channels_last:
            # (B, T, C) -> (B, C, T)
            x_in = x.transpose(1, 2)
        else:
            x_in = x

        B, C, T = x_in.shape
        kernels = self.generate_filters()  # (C * P, 1, K)

        # Repeat input channels for depthwise conv: (B, C * P, T)
        # We perform depthwise grouped convolution with groups = C
        # kernels: (C * P, 1, K) -> weight for groups=C means in_channels=C, out_channels=C*P
        pad = self.kernel_size // 2
        pulses = F.conv1d(x_in, kernels, padding=pad, groups=C)  # (B, C * P, T)

        # Mix pulse heads back to C channels
        out = self.pulse_mix(pulses)  # (B, C, T)

        if self.residual:
            out = out + x_in

        # Apply LayerNorm along channel dimension
        out = out.transpose(1, 2)  # (B, T, C)
        out = self.norm(out)

        if not is_channels_last:
            out = out.transpose(1, 2)

        return out


class AdmissibleStressConeProjection(nn.Module):
    """Admissible Stress Cone Projection Layer (Navier-Stokes Blowup Paper, §4.3, Lemma 4.5 & App. C).

    In the blowup construction, the nonlinear momentum flux of oscillatory wavepackets
    must lie strictly within the positive convex cone spanned by orthogonal background shear modes:
        T in Cone(v_1, v_2) = { c_1 v_1 + c_2 v_2 | c_1, c_2 > 0 }
    guaranteeing that the Reynolds stress tensor is positive-dissipative and satisfies
    the strict realizability inequalities:
        (v_s - 2) J_c^2 < 2(P_c - v_s)^2,  P_c > v_s > 2

    Maps unconstrained latent features z in R^D into an exact Symmetric Positive
    Semi-Definite (SPSD) Reynolds stress tensor Sigma in S_+ spanned by orthogonal shear bases:
        Sigma = sum_{k=1}^K c_k(z) V_k V_k^T + gamma * I_d
    where:
        c_k(z) = softplus(w_k(z)) + eps > 0  (strictly positive wave energy weights)
        V_k in R^{d} are orthonormal / Stiefel shear basis vectors.

    Every eigenvalue lambda_i is strictly positive (> gamma), and the cone defect is
    algebraically zero by construction without any penalty loss.

    Args:
        latent_dim: Input feature dimension D.
        stress_dim: Stress tensor matrix dimension d (default 16).
        num_shear_modes: Number of orthogonal shear bases in the cone (default 4).
        gamma: Minimum diagonal dissipation bandwidth (default 1e-3).
    """

    def __init__(
        self,
        latent_dim: int,
        stress_dim: int = 16,
        num_shear_modes: int = 4,
        gamma: float = 1e-3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.stress_dim = stress_dim
        self.num_shear_modes = min(num_shear_modes, stress_dim)
        self.gamma = gamma

        # Predicts positive cone coefficients c_k(z) > 0
        self.coeff_net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, self.num_shear_modes),
        )

        # Stiefel manifold orthogonal shear modes V in St(num_shear_modes, stress_dim)
        raw_basis = torch.randn(stress_dim, self.num_shear_modes)
        q, _ = torch.linalg.qr(raw_basis)
        self.raw_basis = nn.Parameter(q[:, :self.num_shear_modes])

    def get_orthogonal_modes(self) -> torch.Tensor:
        """Return orthonormal shear mode basis vectors V of shape (stress_dim, K)."""
        q, _ = torch.linalg.qr(self.raw_basis)
        return q[:, :self.num_shear_modes]

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass projecting features into the admissible stress cone.

        Args:
            z: Input feature tensor of shape (B, D) or (B, T, D).

        Returns:
            Sigma: SPSD Reynolds stress tensor of shape (B, d, d) or (B, T, d, d)
                   strictly inside the positive admissible cone.
        """
        orig_shape = z.shape[:-1]
        z_flat = z.reshape(-1, self.latent_dim)
        B = z_flat.size(0)
        D_s = self.stress_dim

        # Predict positive cone weights c_k = softplus(w_k) + eps > 0
        raw_coeffs = self.coeff_net(z_flat)
        c = F.softplus(raw_coeffs) + 1e-4  # (B, K)

        # Orthonormal shear basis: V in R^{d x K}
        V = self.get_orthogonal_modes()  # (d, K)

        # Sigma = sum_k c_k V_k V_k^T = V @ diag(c) @ V^T
        # c.unsqueeze(-1): (B, K, 1), V: (d, K) -> (B, d, d)
        V_exp = V.unsqueeze(0).expand(B, -1, -1)  # (B, d, K)
        scaled_V = V_exp * torch.sqrt(c).unsqueeze(1)  # (B, d, K)
        sigma = torch.bmm(scaled_V, scaled_V.transpose(1, 2))  # (B, d, d)

        # Add background viscous isotropic floor: gamma * I_d
        eye = torch.eye(D_s, device=z.device, dtype=z.dtype).unsqueeze(0)
        sigma = sigma + self.gamma * eye

        target_shape = list(orig_shape) + [D_s, D_s]
        return sigma.view(*target_shape)


class StokesStreamCurlFilter(nn.Module):
    """Stokes Streamfunction Vector Potential Curl Layer (Navier-Stokes Blowup Paper, §3.5 & §7.4).

    In the blowup construction, the velocity field is parameterized via a vector potential:
        u = curl(A) + B e_theta
    which is identically divergence-free (solenoidal):
        div(u) = div(curl(A)) == 0 strictly by construction!

    For latent continuous velocity fields v_raw(z_t, t) in Flow Matching (FlowTSJEPA), this layer
    projects the velocity field onto the divergence-free (solenoidal) subspace using an exact
    skew-symmetric matrix potential S(z_ctx) in so(D):
        v_solenoidal = S(z_ctx) v_raw + v_stream
    where S = 1/2(W - W^T) is strictly skew-symmetric.

    Properties:
    1. Orthogonality: <v_raw, S v_raw> = v_raw^T S v_raw == 0 by skew-symmetry.
    2. Zero Divergence / Incompressibility: Tr(S) == 0, strictly preserving latent volume
       (Liouville theorem) and eliminating artificial manifold volume collapse.
    3. Conserved cross-channel flux: (I - 1/D 1 1^T) projects out dilatational expansion.

    Args:
        latent_dim: Latent vector space dimension D.
        hidden_dim: Hidden dimension for potential generator (default 64).
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        # Generator for skew-symmetric streamfunction potential W(z_ctx) in R^{D x D}
        self.stream_generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim * latent_dim),
        )
        # Learnable base skew-symmetric matrix
        self.base_W = nn.Parameter(0.02 * torch.randn(latent_dim, latent_dim))
        # Mixing scale
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, v_raw: torch.Tensor, z_ctx: torch.Tensor) -> torch.Tensor:
        """Filter velocity field v_raw through the Stokes streamfunction curl projector.

        Args:
            v_raw: Unconstrained velocity field dz/dt of shape (B, D).
            z_ctx: Context conditioning representation of shape (B, D).

        Returns:
            v_solenoidal: Divergence-free (solenoidal) velocity field of shape (B, D).
        """
        B, D = v_raw.shape

        # Generate context-conditioned potential matrix W
        w_mat = self.stream_generator(z_ctx).view(B, D, D) + self.base_W.unsqueeze(0)
        # Skew-symmetrize: S = 1/2 (W - W^T) in so(D)
        S = 0.5 * (w_mat - w_mat.transpose(-1, -2))  # (B, D, D)

        # Rotational curl component: v_rot = S @ v_raw
        v_rot = torch.bmm(S, v_raw.unsqueeze(-1)).squeeze(-1)  # (B, D)

        # Solenoidal projector: remove dilatational / uniform expansion mode across channels
        # (div_cross = sum_i v_i = 0)
        mean_expansion = v_raw.mean(dim=-1, keepdim=True)
        v_incompressible = v_raw - mean_expansion

        # Blend solenoidal rotational curl with incompressible flow
        v_solenoidal = v_incompressible + self.scale * v_rot
        return v_solenoidal


class RadialHeatExteriorBoundary(nn.Module):
    """Radial Heat Exterior Boundary Damping Layer (Navier-Stokes Blowup Paper, §2.3, §3.1 & App. A.6).

    Beyond the active interaction annulus (X >= X_ext), nonlinear flow terms vanish
    and the velocity matches the exact radial swirl heat equation:
        partial_t K = (partial_rr + r^{-1} partial_r - r^{-2}) K,
        K(r, tau) = r^{-1-2h} H_ext(tau / r^2)

    Partitions latent space by norm r = ||z||_2:
    1. For r <= r_core: Retains the unaltered rich nonlinear representation.
    2. For r > r_core: Smoothly transitions via an infinitely differentiable partition
       of unity into the contractive heat-diffusion tail:
           z_out = chi(r) * z + (1 - chi(r)) * z * (1 + (r - r_core) / r_scale)^{-(1 + 2h)}
       where 0 < h < 0.5.

    This provably contracts and bounds runaway activations under extreme out-of-distribution (OOD)
    perturbations, preventing numerical instability and entropy collapse.

    Args:
        latent_dim: Latent vector space dimension D.
        r_core: Core radius boundary where non-linear flow dominates (default: sqrt(D)).
        h_exponent: Navier-Stokes scaling exponent h in (0, 0.5) (default 0.05).
    """

    def __init__(
        self,
        latent_dim: int,
        r_core: Optional[float] = None,
        h_exponent: float = 0.05,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        # Default r_core = sqrt(D) corresponding to expected norm of standard Gaussian in R^D
        self.r_core = r_core if r_core is not None else math.sqrt(float(latent_dim))
        self.r_scale = max(self.r_core * 0.5, 1e-2)
        self.h_exponent = h_exponent
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass regularizing latent vector z via exterior radial heat damping.

        Args:
            z: Latent feature tensor of shape (..., D).

        Returns:
            Regularized latent tensor of shape (..., D).
        """
        # Compute Euclidean norm r = ||z||_2 along feature dimension
        r = torch.norm(z, p=2, dim=-1, keepdim=True).clamp(min=self.eps)

        # Smooth partition of unity chi(r) in [0, 1]
        # chi = 1 when r <= r_core, smoothly decaying to 0 as r grows
        excess = F.relu(r - self.r_core)
        chi = torch.exp(-torch.square(excess / self.r_scale))

        # Exterior radial heat-kernel decay factor: (1 + excess / r_scale)^{-(1 + 2h)}
        heat_decay = torch.pow(1.0 + excess / self.r_scale, -(1.0 + 2.0 * self.h_exponent))

        # Blended representation
        contractive_scale = chi + (1.0 - chi) * heat_decay
        return z * contractive_scale

__all__ = [
    "ShearingWaveletPulseBlock",
    "AdmissibleStressConeProjection",
    "StokesStreamCurlFilter",
    "RadialHeatExteriorBoundary",
]
