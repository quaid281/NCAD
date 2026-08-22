"""Sparse Identification of Nonlinear Dynamics (SINDy) for NCAD-CS.

Discovers explicit non-linear differential equations z_dot = Theta(z) * Xi
governing normal latent state transitions. Dynamical residual scores R(t)
measure physical consistency and flag non-physical state transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SINDyConfig:
    poly_degree: int = 2
    include_constant: bool = True
    threshold: float = 0.05
    alpha: float = 1e-4
    max_iter: int = 10


class SINDyDynamicsScorer:
    """Discovers governing equations z_dot = Theta(z) * Xi and computes residual scores."""

    def __init__(self, config: Optional[SINDyConfig] = None):
        self.config = config or SINDyConfig()
        self.coefficients: Optional[np.ndarray] = None
        self.latent_dim: int = 0
        self.n_features: int = 0

    def build_library(self, z: np.ndarray) -> np.ndarray:
        """Construct polynomial feature library Theta(Z) up to poly_degree."""
        z = np.asarray(z, dtype=np.float64)
        if z.ndim == 1:
            z = z.reshape(1, -1)
        n_samples, n_dim = z.shape

        cols = []
        if self.config.include_constant:
            cols.append(np.ones((n_samples, 1), dtype=np.float64))

        # Degree 1
        cols.append(z)

        # Degree 2
        if self.config.poly_degree >= 2:
            deg2_cols = []
            for i in range(n_dim):
                for j in range(i, n_dim):
                    deg2_cols.append((z[:, i] * z[:, j])[:, None])
            if deg2_cols:
                cols.append(np.hstack(deg2_cols))

        return np.hstack(cols)

    def fit(self, z_sequences: np.ndarray, dt: float = 1.0) -> "SINDyDynamicsScorer":
        """Fit sparse governing coefficient matrix Xi using STLSQ."""
        z_seq = np.asarray(z_sequences, dtype=np.float64)
        if z_seq.ndim != 2:
            raise ValueError("Expected 2D array of shape (n_samples, latent_dim).")

        n_samples, latent_dim = z_seq.shape
        if n_samples < 5:
            raise ValueError("Insufficient samples to fit SINDy dynamics.")

        self.latent_dim = latent_dim

        # Compute numerical time derivatives z_dot via 2nd-order central differences
        z_dot = np.zeros_like(z_seq)
        z_dot[1:-1] = (z_seq[2:] - z_seq[:-2]) / (2.0 * dt)
        z_dot[0] = (z_seq[1] - z_seq[0]) / dt
        z_dot[-1] = (z_seq[-1] - z_seq[-2]) / dt

        theta = self.build_library(z_seq)
        self.n_features = theta.shape[1]

        # STLSQ (Sequential Thresholded Least Squares)
        xi = np.linalg.lstsq(
            theta.T @ theta + self.config.alpha * np.eye(self.n_features),
            theta.T @ z_dot,
            rcond=None,
        )[0]

        for _ in range(self.config.max_iter):
            small_inds = np.abs(xi) < self.config.threshold
            xi[small_inds] = 0.0
            for d in range(latent_dim):
                big_inds = ~small_inds[:, d]
                if np.any(big_inds):
                    theta_sub = theta[:, big_inds]
                    xi[big_inds, d] = np.linalg.lstsq(
                        theta_sub.T @ theta_sub + self.config.alpha * np.eye(np.sum(big_inds)),
                        theta_sub.T @ z_dot[:, d],
                        rcond=None,
                    )[0]

        self.coefficients = xi.astype(np.float32)
        return self

    def score(self, z_sequences: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """Compute dynamical physics residual score R(t) = ||z_dot - Theta(z) * Xi||_2."""
        if self.coefficients is None:
            raise RuntimeError("SINDyDynamicsScorer must be fitted before calling score().")

        z_seq = np.asarray(z_sequences, dtype=np.float64)
        if z_seq.ndim != 2:
            raise ValueError("Expected 2D array of shape (n_samples, latent_dim).")

        n_samples, latent_dim = z_seq.shape
        z_dot = np.zeros_like(z_seq)
        if n_samples >= 3:
            z_dot[1:-1] = (z_seq[2:] - z_seq[:-2]) / (2.0 * dt)
            z_dot[0] = (z_seq[1] - z_seq[0]) / dt
            z_dot[-1] = (z_seq[-1] - z_seq[-2]) / dt

        theta = self.build_library(z_seq)
        z_dot_pred = theta @ self.coefficients
        residuals = np.linalg.norm(z_dot - z_dot_pred, axis=1)
        return residuals.astype(np.float32)
