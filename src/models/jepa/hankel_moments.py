"""Hankel Power-Sum Moment Discriminant for Regime Dynamics (Chapter 7 of Ten Advances).

Reconstructs and monitors multi-modal operational regimes using permutation-invariant
power-sum moment sequences and the algebraic Hankel determinant:
    Delta_K = det( M_{i+l} )_{0 <= i, l < K} = prod_{1 <= r < s <= K} (z_s - z_r)^2

Provides:
1. Permutation-invariant tracking of K discrete operating regimes without cluster assignment.
2. Exact Hankel discriminant barrier preventing regime collapse (Delta_K -> 0).
3. Higher-moment discrepancy scoring detecting non-Gaussian state transitions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HankelMomentTracker(nn.Module):
    """Algebraic Hankel Moment Tracker for Operational Regimes.
    
    Computes power-sum moments M_j = sum_{k=1}^K z_k^j and their Hankel matrix
    determinant to quantify mode separation and detect regime collapse.
    """

    def __init__(self, n_regimes: int = 4, max_moment: Optional[int] = None, eps: float = 1e-6):
        super().__init__()
        self.n_regimes = n_regimes
        # Need up to moment 2*K - 2 to form a K x K Hankel matrix
        self.K = n_regimes
        self.max_moment = max_moment if max_moment is not None else (2 * n_regimes - 2)
        self.eps = eps

    def compute_power_sum_moments(self, regimes: torch.Tensor) -> torch.Tensor:
        """Compute coordinate-wise power-sum moments across K regimes.
        
        Args:
            regimes: Tensor of shape (..., K, D) representing K regime vectors.
            
        Returns:
            moments: Tensor of shape (..., 2K - 1, D) with moments M_0, ..., M_{2K-2}.
        """
        moments = []
        for j in range(2 * self.K - 1):
            if j == 0:
                # M_0 = K * ones
                m_j = torch.full_like(regimes[..., 0, :], float(self.K))
            else:
                # M_j = sum_{k=1}^K z_k^j
                m_j = torch.sum(torch.pow(regimes, j), dim=-2)
            moments.append(m_j)
        return torch.stack(moments, dim=-2)  # (..., 2K-1, D)

    def build_hankel_matrix(self, moments: torch.Tensor) -> torch.Tensor:
        """Construct K x K Hankel matrix from power-sum moments for each latent dimension.
        
        Args:
            moments: Tensor of shape (..., 2K - 1, D).
            
        Returns:
            hankel: Tensor of shape (..., D, K, K) where H_{i,l} = M_{i+l}.
        """
        K = self.K
        # Transpose to put D before moments: (..., D, 2K - 1)
        m_trans = moments.transpose(-2, -1)
        rows = []
        for i in range(K):
            row_i = m_trans[..., i : i + K]  # (..., D, K)
            rows.append(row_i)
        hankel = torch.stack(rows, dim=-2)  # (..., D, K, K)
        return hankel

    def compute_hankel_determinant(self, regimes: torch.Tensor) -> torch.Tensor:
        """Compute log Hankel determinant across latent dimensions.
        
        A non-zero Hankel determinant certifies that all K regimes are mutually distinct.
        Collapse of any two regimes forces Delta_K -> 0.
        
        Args:
            regimes: Tensor of shape (..., K, D).
            
        Returns:
            log_det: Tensor of shape (..., D) containing log|Delta_K|.
        """
        moments = self.compute_power_sum_moments(regimes)
        hankel = self.build_hankel_matrix(moments)  # (..., D, K, K)
        # Slogdet for numerical stability
        sign, log_abs_det = torch.linalg.slogdet(hankel + self.eps * torch.eye(self.K, device=regimes.device))
        return log_abs_det

    def compute_moment_discrepancy(self, z_pred: torch.Tensor, z_tgt: torch.Tensor) -> torch.Tensor:
        """Compute multi-order moment discrepancy between predicted and target representations.
        
        Instead of only measuring L2 distance (first-order moment), measures higher-order
        moment matching up to degree 4.
        
        Args:
            z_pred: (B, D) predicted representations.
            z_tgt: (B, D) target representations.
            
        Returns:
            discrepancy: (B,) scalar discrepancy score.
        """
        disc = torch.zeros(z_pred.size(0), device=z_pred.device, dtype=z_pred.dtype)
        weights = [1.0, 0.5, 0.25, 0.125]
        for p in range(1, len(weights) + 1):
            diff_p = torch.pow(z_pred, p) - torch.pow(z_tgt, p)
            disc = disc + weights[p - 1] * torch.linalg.norm(diff_p, dim=-1)
        return disc

    def compute_regime_separation_loss(self, regimes: torch.Tensor, min_log_det: float = -5.0) -> torch.Tensor:
        """Compute algebraic separation loss penalizing regime collapse via Hankel determinant.
        
        Args:
            regimes: Tensor of shape (K, D) representing learnable operating regimes.
            min_log_det: Minimum acceptable log-determinant barrier.
            
        Returns:
            loss: Scalar penalty >= 0.
        """
        log_det = self.compute_hankel_determinant(regimes)  # (D,)
        # Penalize if log_det falls below min_log_det
        barrier = F.relu(min_log_det - log_det)
        return barrier.mean()
