"""Layer Activation Tracer and Diagnostic Autopsy Suite for NCAD-CS models.

Provides non-invasive runtime inspection, numerical health auditing, dimensional
collapse detection, and perturbation divergence tracking across all PyTorch
neural architectures in the repository.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


@dataclass
class LayerTraceRecord:
    """Diagnostic profile for a single layer or submodule execution."""

    name: str
    module_type: str
    input_shapes: List[Tuple[int, ...]]
    output_shape: Tuple[int, ...]
    mean: float
    std: float
    norm: float
    min_val: float
    max_val: float
    sparsity: float
    stable_rank: float
    batch_variance: float
    health_flag: str
    tensor: Optional[torch.Tensor] = None


@dataclass
class AutopsyReport:
    """Complete diagnostic autopsy for a neural model."""

    model_name: str
    records: List[LayerTraceRecord]
    global_health: str
    collapsed_layers: List[str]
    dead_layers: List[str]
    unstable_layers: List[str]
    sequential_drift: Dict[str, float] = field(default_factory=dict)
    anomaly_divergence: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a human-readable text summary of the autopsy."""
        lines = [
            f"=== Autopsy Report for {self.model_name} ===",
            f"Total Submodules Traced: {len(self.records)}",
            f"Global Health Status:    {self.global_health}",
            f"Collapsed Layers:        {len(self.collapsed_layers)} ({', '.join(self.collapsed_layers[:3]) + ('...' if len(self.collapsed_layers) > 3 else '') if self.collapsed_layers else 'None'})",
            f"Dead/High Sparsity:      {len(self.dead_layers)} ({', '.join(self.dead_layers[:3]) + ('...' if len(self.dead_layers) > 3 else '') if self.dead_layers else 'None'})",
            f"Unstable (NaN/Inf/Expl): {len(self.unstable_layers)} ({', '.join(self.unstable_layers[:3]) + ('...' if len(self.unstable_layers) > 3 else '') if self.unstable_layers else 'None'})",
        ]
        if self.anomaly_divergence:
            max_div_layer = max(self.anomaly_divergence.items(), key=lambda x: x[1])
            lines.append(f"Peak Anomaly Divergence: {max_div_layer[0]} (gain: {max_div_layer[1]:.4f})")
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert layer trace records into a pandas DataFrame."""
        data = []
        for r in self.records:
            row = {
                "layer_name": r.name,
                "module_type": r.module_type,
                "output_shape": str(r.output_shape),
                "mean": r.mean,
                "std": r.std,
                "l2_norm": r.norm,
                "sparsity": r.sparsity,
                "stable_rank": r.stable_rank,
                "batch_variance": r.batch_variance,
                "health_flag": r.health_flag,
            }
            if r.name in self.sequential_drift:
                row["sequential_cosine_sim"] = self.sequential_drift[r.name]
            if r.name in self.anomaly_divergence:
                row["anomaly_divergence"] = self.anomaly_divergence[r.name]
            data.append(row)
        return pd.DataFrame(data)

    def markdown_table(self, max_rows: int = 25) -> str:
        """Generate a GitHub-flavored markdown summary table."""
        df = self.to_dataframe()
        if len(df) == 0:
            return "_No layers traced._"

        display_df = df.head(max_rows)
        cols = ["layer_name", "module_type", "output_shape", "l2_norm", "sparsity", "stable_rank", "health_flag"]
        if "anomaly_divergence" in df.columns:
            cols.append("anomaly_divergence")

        present_cols = [c for c in cols if c in display_df.columns]
        header = "| " + " | ".join(present_cols) + " |"
        sep = "| " + " | ".join(["---"] * len(present_cols)) + " |"
        rows = []
        for _, row in display_df[present_cols].iterrows():
            formatted = []
            for c in present_cols:
                v = row[c]
                if isinstance(v, float):
                    formatted.append(f"{v:.4f}")
                else:
                    formatted.append(str(v))
            rows.append("| " + " | ".join(formatted) + " |")

        table = "\n".join([header, sep] + rows)
        if len(df) > max_rows:
            table += f"\n\n_... ({len(df) - max_rows} additional layers truncated)_"
        return table


def _compute_stable_rank(tensor: torch.Tensor) -> float:
    """Compute numerical stable rank: ||A||_F^2 / ||A||_2^2 on 2D flattened representation."""
    if tensor.numel() == 0:
        return 0.0
    # Flatten to (batch_size, -1) or 2D matrix
    mat = tensor.detach().float()
    if mat.dim() > 2:
        mat = mat.reshape(mat.shape[0], -1)
    elif mat.dim() == 1:
        mat = mat.unsqueeze(0)

    # If matrix has single row/col or non-finite elements
    if mat.shape[0] < 2 or mat.shape[1] < 2 or not torch.isfinite(mat).all():
        return 1.0

    try:
        # Center to avoid mean artifact
        mat_centered = mat - mat.mean(dim=0, keepdim=True)
        f_norm_sq = torch.sum(mat_centered**2).item()
        if f_norm_sq < 1e-12:
            return 0.0
        # Operator norm (top singular value)
        top_s = torch.linalg.matrix_norm(mat_centered, ord=2).item()
        if top_s < 1e-8:
            return 0.0
        rank = f_norm_sq / (top_s**2)
        return float(rank)
    except Exception:
        return 1.0


def _compute_batch_variance(tensor: torch.Tensor) -> float:
    """Compute average channel variance across the batch dimension."""
    if tensor.numel() == 0 or tensor.shape[0] <= 1:
        return 0.0
    detached = tensor.detach().float()
    if not torch.isfinite(detached).all():
        return 0.0
    # Variance along batch dim 0, averaged over remaining dims
    var_per_elem = torch.var(detached, dim=0, unbiased=False)
    return float(torch.mean(var_per_elem).item())


class LayerActivationTracer:
    """Context manager for non-invasive layer-to-layer activation tracing."""

    def __init__(
        self,
        model: nn.Module,
        target_classes: Optional[Sequence[type]] = None,
        exclude_classes: Optional[Sequence[type]] = (nn.Identity, nn.Dropout, nn.Dropout1d, nn.Dropout2d),
        leaf_only: bool = True,
        capture_tensors: bool = True,
        cpu_offload: bool = True,
    ):
        """Initialize the tracer.

        Parameters
        ----------
        model : nn.Module
            The model to trace.
        target_classes : Optional[Sequence[type]]
            If provided, only modules of these types are hooked.
        exclude_classes : Optional[Sequence[type]]
            Modules of these types are ignored.
        leaf_only : bool
            If True, only hooks leaf submodules (modules with no child modules).
        capture_tensors : bool
            If True, stores detached activation tensors for geometric comparisons.
        cpu_offload : bool
            If True, moves captured tensors to CPU immediately to avoid VRAM bloat.
        """
        self.model = model
        self.target_classes = tuple(target_classes) if target_classes else None
        self.exclude_classes = tuple(exclude_classes) if exclude_classes else ()
        self.leaf_only = leaf_only
        self.capture_tensors = capture_tensors
        self.cpu_offload = cpu_offload

        self.records: OrderedDict[str, LayerTraceRecord] = OrderedDict()
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._execution_counter: int = 0

    def __enter__(self) -> "LayerActivationTracer":
        self.clear()
        self._register_hooks()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.remove_hooks()

    def clear(self) -> None:
        """Clear recorded traces and reset counter."""
        self.records.clear()
        self._execution_counter = 0

    def remove_hooks(self) -> None:
        """Remove all active PyTorch forward hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _should_hook(self, name: str, module: nn.Module) -> bool:
        if name == "":
            return False
        if isinstance(module, self.exclude_classes):
            return False
        if self.leaf_only and len(list(module.children())) > 0:
            return False
        if self.target_classes is not None:
            return isinstance(module, self.target_classes)
        return True

    def _register_hooks(self) -> None:
        for name, module in self.model.named_modules():
            if self._should_hook(name, module):
                handle = module.register_forward_hook(self._make_hook(name, type(module).__name__))
                self._handles.append(handle)

    def _make_hook(self, layer_name: str, module_type: str) -> Callable:
        def hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
            # Extract primary tensor output
            act_tensor: Optional[torch.Tensor] = None
            if isinstance(output, torch.Tensor):
                act_tensor = output
            elif isinstance(output, (tuple, list)) and len(output) > 0:
                for item in output:
                    if isinstance(item, torch.Tensor):
                        act_tensor = item
                        break
            elif isinstance(output, dict) and len(output) > 0:
                for item in output.values():
                    if isinstance(item, torch.Tensor):
                        act_tensor = item
                        break

            if act_tensor is None or act_tensor.numel() == 0:
                return

            # Extract input shapes
            input_shapes = []
            for inp in inputs:
                if isinstance(inp, torch.Tensor):
                    input_shapes.append(tuple(inp.shape))

            output_shape = tuple(act_tensor.shape)
            detached = act_tensor.detach()
            if self.cpu_offload:
                detached = detached.cpu()

            # Compute numerical health metrics
            has_nan = torch.isnan(detached).any().item()
            has_inf = torch.isinf(detached).any().item()

            if has_nan or has_inf:
                mean_val = float("nan")
                std_val = float("nan")
                norm_val = float("nan")
                min_val = float("nan")
                max_val = float("nan")
                sparsity = 1.0
                stable_rank = 0.0
                batch_var = 0.0
                health_flag = "UNSTABLE_NAN_INF"
            else:
                float_t = detached.float()
                mean_val = float(float_t.mean().item())
                std_val = float(float_t.std().item()) if float_t.numel() > 1 else 0.0
                norm_val = float(torch.linalg.norm(float_t).item())
                min_val = float(float_t.min().item())
                max_val = float(float_t.max().item())

                # Zero/near-zero elements
                zero_mask = torch.abs(float_t) <= 1e-7
                sparsity = float((zero_mask.sum().item()) / float_t.numel())
                stable_rank = _compute_stable_rank(float_t)
                batch_var = _compute_batch_variance(float_t)

                # Determine health flag
                if norm_val > 1e4:
                    health_flag = "EXPLODING"
                elif norm_val < 1e-7:
                    health_flag = "VANISHING"
                elif batch_var < 1e-7 and float_t.shape[0] > 1:
                    health_flag = "COLLAPSED"
                elif sparsity >= 0.95:
                    health_flag = "HIGH_SPARSITY"
                else:
                    health_flag = "HEALTHY"

            # In case a module is called multiple times in a forward pass (e.g. recurrent or shared weights),
            # uniquely index it.
            record_key = layer_name
            if record_key in self.records:
                self._execution_counter += 1
                record_key = f"{layer_name}#{self._execution_counter}"

            record = LayerTraceRecord(
                name=record_key,
                module_type=module_type,
                input_shapes=input_shapes,
                output_shape=output_shape,
                mean=mean_val,
                std=std_val,
                norm=norm_val,
                min_val=min_val,
                max_val=max_val,
                sparsity=sparsity,
                stable_rank=stable_rank,
                batch_variance=batch_var,
                health_flag=health_flag,
                tensor=detached if self.capture_tensors else None,
            )
            self.records[record_key] = record

        return hook


class LayerAutopsy:
    """Diagnostic autopsy orchestrator evaluating layer dynamics and representation health."""

    @staticmethod
    def inspect(
        nominal_tracer: LayerActivationTracer,
        anomaly_tracer: Optional[LayerActivationTracer] = None,
        model_name: str = "Model",
    ) -> AutopsyReport:
        """Analyze activation traces from nominal and optional anomalous forward passes."""
        records = list(nominal_tracer.records.values())

        collapsed_layers = [r.name for r in records if r.health_flag == "COLLAPSED"]
        dead_layers = [r.name for r in records if r.health_flag == "HIGH_SPARSITY"]
        unstable_layers = [r.name for r in records if r.health_flag in ("UNSTABLE_NAN_INF", "EXPLODING", "VANISHING")]

        # Determine global health
        if unstable_layers:
            global_health = "CRITICAL_UNSTABLE"
        elif collapsed_layers:
            global_health = "WARNING_COLLAPSE"
        elif len(dead_layers) > len(records) * 0.5:
            global_health = "WARNING_HIGH_DEAD_NEURONS"
        else:
            global_health = "HEALTHY"

        # 1. Compute Sequential Drift (cosine similarity between adjacent layers with matching feature dims)
        sequential_drift: Dict[str, float] = {}
        for i in range(len(records) - 1):
            curr_r = records[i]
            next_r = records[i + 1]
            if curr_r.tensor is not None and next_r.tensor is not None:
                # If shapes can be compared or flattened
                curr_flat = curr_r.tensor.float().reshape(curr_r.tensor.shape[0], -1)
                next_flat = next_r.tensor.float().reshape(next_r.tensor.shape[0], -1)
                if curr_flat.shape == next_flat.shape and curr_flat.shape[1] > 0:
                    sim = torch.nn.functional.cosine_similarity(curr_flat, next_flat, dim=1).mean().item()
                    sequential_drift[f"{curr_r.name} -> {next_r.name}"] = float(sim)

        # 2. Compute Anomaly Divergence (if anomalous trace is provided)
        anomaly_divergence: Dict[str, float] = {}
        if anomaly_tracer is not None:
            for r in records:
                if r.name in anomaly_tracer.records:
                    nom_r = r
                    anom_r = anomaly_tracer.records[r.name]
                    if nom_r.tensor is not None and anom_r.tensor is not None:
                        nom_t = nom_r.tensor.float()
                        anom_t = anom_r.tensor.float()
                        if nom_t.shape == anom_t.shape:
                            diff_norm = torch.linalg.norm(anom_t - nom_t).item()
                            denom = torch.linalg.norm(nom_t).item() + 1e-8
                            divergence = diff_norm / denom
                            anomaly_divergence[r.name] = float(divergence)

        return AutopsyReport(
            model_name=model_name,
            records=records,
            global_health=global_health,
            collapsed_layers=collapsed_layers,
            dead_layers=dead_layers,
            unstable_layers=unstable_layers,
            sequential_drift=sequential_drift,
            anomaly_divergence=anomaly_divergence,
        )


def autopsy_model(
    model: nn.Module,
    sample_input: Union[torch.Tensor, Sequence[torch.Tensor]],
    anomalous_input: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
    model_name: Optional[str] = None,
    leaf_only: bool = True,
    target_classes: Optional[Sequence[type]] = None,
) -> AutopsyReport:
    """Run an automated diagnostic autopsy on a model instance.

    Parameters
    ----------
    model : nn.Module
        The PyTorch neural network model.
    sample_input : Tensor or tuple of Tensors
        Nominal background input (e.g. (context, target) or single input x).
    anomalous_input : Optional Tensor or tuple of Tensors
        Anomalous counterpart input to trace perturbation propagation.
    model_name : Optional str
        Name for reporting.
    leaf_only : bool
        Whether to trace leaf submodules.
    target_classes : Optional Sequence of types
        Specific module classes to hook.
    """
    name = model_name or type(model).__name__
    model.eval()

    def run_forward(inp: Union[torch.Tensor, Sequence[torch.Tensor]]) -> Any:
        with torch.no_grad():
            if isinstance(inp, (tuple, list)):
                return model(*inp)
            return model(inp)

    # 1. Run nominal trace
    with LayerActivationTracer(
        model,
        target_classes=target_classes,
        leaf_only=leaf_only,
        capture_tensors=True,
    ) as nom_tracer:
        run_forward(sample_input)

    # 2. Run anomalous trace (if provided)
    anom_tracer: Optional[LayerActivationTracer] = None
    if anomalous_input is not None:
        with LayerActivationTracer(
            model,
            target_classes=target_classes,
            leaf_only=leaf_only,
            capture_tensors=True,
        ) as a_tracer:
            run_forward(anomalous_input)
        anom_tracer = a_tracer

    # 3. Compile autopsy report
    return LayerAutopsy.inspect(nominal_tracer=nom_tracer, anomaly_tracer=anom_tracer, model_name=name)
