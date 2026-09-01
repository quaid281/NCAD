"""Shared utilities for JEPA-family models.

Provides helpers used by all JEPA variants to avoid duplicated lifecycle code
and to support memory-efficient batched covariance fitting from numpy arrays.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

ArrayLike = Union[np.ndarray, torch.Tensor]


class JEPABase(nn.Module):
    """Mixin/base class centralising shared JEPA lifecycle behaviour.

    Subclasses are expected to:
      * set ``self.context_encoder`` and ``self.target_encoder``
      * set ``self.ema_decay``
      * register ``precision_matrix``, ``residual_mean``, ``precision_fitted``
        buffers (or call :meth:`register_mahalanobis_buffers`)

    This base class provides:
      * :meth:`init_target_encoder` — deep-copy, freeze, eval-mode the target
      * :meth:`train` override keeping the target encoder in eval mode
      * :meth:`update_target_encoder` — EMA parameter + buffer sync
    """

    def init_target_encoder(self, context_encoder: nn.Module) -> nn.Module:
        """Deep-copy *context_encoder*, freeze it, and force eval mode."""
        import copy

        target = copy.deepcopy(context_encoder)
        for p in target.parameters():
            p.requires_grad = False
        target.eval()
        return target

    def register_mahalanobis_buffers(self, dim: int) -> None:
        """Register the standard Mahalanobis precision/mean/fitted buffers."""
        self.register_buffer("precision_matrix", torch.eye(dim))
        self.register_buffer("residual_mean", torch.zeros(dim))
        self.register_buffer("precision_fitted", torch.tensor(False, dtype=torch.bool))

    def train(self, mode: bool = True) -> "JEPABase":
        """Override train to keep the target encoder strictly in eval mode."""
        super().train(mode)
        if hasattr(self, "target_encoder"):
            self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target_encoder(self, decay: Optional[float] = None) -> None:
        """EMA update of target encoder parameters and buffers."""
        m = self.ema_decay if decay is None else decay
        for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(m).add_((1.0 - m) * param_q.data)
        for buf_q, buf_k in zip(self.context_encoder.buffers(), self.target_encoder.buffers()):
            buf_k.data.copy_(buf_q.data)

    def compute_objective(
        self,
        ctx: torch.Tensor,
        tgt: torch.Tensor,
        config,
        *,
        injector=None,
        full_batch=None,
    ):
        """Compute the training/validation objective for this JEPA variant.

        Subclasses must override this to return ``(total_loss, metrics_dict)``.
        Centralising the objective here lets the trainer dispatch polymorphically
        instead of maintaining a per-class ``isinstance`` chain.

        Args:
            ctx: (B, L_ctx, C) context windows.
            tgt: (B, L_tgt, C) target windows.
            config: ``CSMConfig`` with VICReg weights and other hyperparameters.
            injector: Optional ``ContextualAnomalyInjector`` for NCAD-JEPA.
            full_batch: Optional (B, L_full, C) numpy array for NCAD-JEPA injection.

        Returns:
            Tuple of (total_loss_tensor, metrics_dict).
        """
        raise NotImplementedError(f"{type(self).__name__} must implement compute_objective")


def to_device_tensor(arr: ArrayLike, model: nn.Module, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert a numpy array or tensor to a float tensor on the model's device.

    This enables ``fit_mahalanobis_covariance`` and scoring methods to accept
    numpy arrays directly, avoiding the need for callers to transfer the entire
    dataset to GPU upfront.
    """
    if isinstance(arr, np.ndarray):
        return torch.from_numpy(arr).to(dtype=dtype, device=_model_device(model))
    return arr.to(device=_model_device(model), dtype=dtype)


def _model_device(model: nn.Module) -> torch.device:
    """Return the device of the first parameter of *model*."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def fit_covariance_batched(
    model: nn.Module,
    context_windows: ArrayLike,
    target_windows: ArrayLike,
    *,
    residual_fn,
    dim: int,
    batch_size: int = 512,
    reg: float = 1e-3,
    precision_buffer: torch.Tensor,
    residual_mean_buffer: torch.Tensor,
    fitted_buffer: torch.Tensor,
) -> None:
    """Fit empirical residual covariance via batched GPU accumulation.

    *residual_fn* receives per-batch context and target tensors (already on the
    model's device) and must return a 2-D residual tensor ``(N_batch, dim)``.

    This avoids materialising the entire residual matrix in GPU memory at once
    by accumulating the covariance sum and count incrementally.
    """
    model.eval()
    n_samples = len(context_windows)
    device = _model_device(model)

    # Incremental covariance accumulation: sum and outer-product sum
    residual_sum = torch.zeros(dim, device=device, dtype=torch.float32)
    outer_sum = torch.zeros((dim, dim), device=device, dtype=torch.float32)
    total_count = 0

    for i in range(0, n_samples, batch_size):
        ctx_b = to_device_tensor(context_windows[i : i + batch_size], model)
        tgt_b = to_device_tensor(target_windows[i : i + batch_size], model)
        residuals = residual_fn(ctx_b, tgt_b).reshape(-1, dim).to(torch.float32)
        residual_sum += residuals.sum(dim=0)
        outer_sum += residuals.T @ residuals
        total_count += residuals.shape[0]

    if total_count == 0:
        raise RuntimeError("Cannot fit covariance from zero samples.")

    mean_res = residual_sum / total_count
    cov = (outer_sum - total_count * torch.outer(mean_res, mean_res)) / max(total_count - 1, 1)
    cov_reg = cov + reg * torch.eye(dim, device=device)
    precision = torch.linalg.pinv(cov_reg)

    residual_mean_buffer.copy_(mean_res)
    precision_buffer.copy_(precision)
    fitted_buffer.copy_(torch.tensor(True, dtype=torch.bool))
