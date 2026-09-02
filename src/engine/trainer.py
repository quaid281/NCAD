"""Training routines and encoder builders for NCAD-CS."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.config import CSMConfig
from src.models.encoders.multi_scale_tcn_encoder import MultiScaleTCNEncoder
from src.models.encoders.relational_gat_encoder import RelationalGATEncoder
from src.models.encoders.selective_ssm_encoder import SelectiveSSMContextEncoder
from src.models.encoders.tcn_encoder import HybridTCNEncoder, contrastive_loss
from src.models.jepa.causal_ssm_flow_jepa import CausalSSMFlowJEPA
from src.models.jepa.flow_ts_jepa import FlowTSJEPAModel
from src.models.jepa.gat_jepa import RelationalGAT_JEPAModel
from src.models.jepa.multiscale_ts_jepa import MultiScaleTSJEPA
from src.models.jepa.ncad_flow_jepa import NCADFlowJEPAModel
from src.models.jepa.ncad_jepa import NCADJEPAModel
from src.models.jepa.patch_flow_jepa import PatchFlowJEPA
from src.models.jepa.patch_ts_jepa import PatchTSJEPA
from src.models.jepa.ts_jepa import TSJEPAModel
from src.models.losses.anomaly_injector import AnomalyInjectionConfig, ContextualAnomalyInjector

logger = logging.getLogger("NCAD.engine.trainer")

EncoderModel = (
    HybridTCNEncoder
    | MultiScaleTCNEncoder
    | RelationalGATEncoder
    | SelectiveSSMContextEncoder
    | TSJEPAModel
    | PatchTSJEPA
    | RelationalGAT_JEPAModel
    | FlowTSJEPAModel
    | PatchFlowJEPA
    | MultiScaleTSJEPA
    | NCADJEPAModel
    | NCADFlowJEPAModel
    | CausalSSMFlowJEPA
)


def set_seed(seed: int) -> None:
    """Set global seeds for reproducibility across NumPy and PyTorch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(device_name: str) -> torch.device:
    """Resolve device string to torch.device with CUDA auto-detection."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_encoder(config: CSMConfig, input_dim: int, device: torch.device) -> nn.Module:
    """Instantiate and initialize the requested temporal sequence encoder."""
    common_kwargs = {
        "input_dim": input_dim,
        "latent_dim": config.latent_dim,
        "filters": config.filters,
        "tcn_layers": config.tcn_layers,
        "kernel_size": config.kernel_size,
        "dropout": config.dropout,
    }
    if config.encoder_architecture == "hybrid_tcn":
        model = HybridTCNEncoder(**common_kwargs)
    elif config.encoder_architecture == "multi_scale_tcn":
        model = MultiScaleTCNEncoder(**common_kwargs)
    elif config.encoder_architecture == "relational_gat":
        model = RelationalGATEncoder(
            input_dim=input_dim,
            latent_dim=config.latent_dim,
            filters=config.filters,
            tcn_layers=config.tcn_layers,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
        )
    elif config.encoder_architecture in ["selective_ssm", "ssm"]:
        model = SelectiveSSMContextEncoder(
            input_dim=input_dim,
            latent_dim=config.latent_dim,
            hidden_dim=max(64, config.latent_dim * 2),
            layers=config.tcn_layers,
            dropout=config.dropout,
        )
    else:
        raise ValueError(f"Unknown encoder architecture: {config.encoder_architecture}")
    model.architecture = config.encoder_architecture
    return model.to(device)


def build_ts_jepa_model(config: CSMConfig, input_dim: int, device: torch.device) -> nn.Module:
    """Instantiate and initialize the requested TS-JEPA model variant."""
    from src.models.registry import canonical_model_type

    canonical = canonical_model_type(config.model_type)
    if canonical == "ts_jepa":
        encoder = build_encoder(config, input_dim, device)
        model = TSJEPAModel(
            context_encoder=encoder,
            latent_dim=config.latent_dim,
            predictor_hidden_dim=max(64, config.latent_dim * 2),
            predictor_layers=2,
            ema_decay=0.996,
            dropout=config.dropout,
        )
    elif canonical == "patch_ts_jepa":
        patch_size = config.patch_size
        n_tgt_patches = max(1, config.suspect_size // patch_size)
        model = PatchTSJEPA(
            input_dim=input_dim,
            patch_size=patch_size,
            d_model=config.filters,
            n_heads=4,
            n_layers=2,
            d_ff=config.filters * 2,
            n_target_patches=n_tgt_patches,
            ema_decay=0.996,
            dropout=config.dropout,
        )
    elif canonical == "gat_jepa":
        model = RelationalGAT_JEPAModel(
            input_dim=input_dim,
            latent_dim=config.latent_dim,
            filters=config.filters,
            tcn_layers=config.tcn_layers,
            gat_layers=2,
            gat_heads=4,
            dropout=config.dropout,
            ema_decay=0.996,
        )
    elif canonical == "ncad":
        # Legacy NCAD uses the contrastive TCN encoder path (is_jepa=False).
        # build_ts_jepa_model is only called for JEPA models, so this branch
        # should never be reached. If it is, redirect to the contrastive path.
        raise ValueError(
            "The 'ncad' model type uses the contrastive encoder path, not build_ts_jepa_model. "
            "Use 'ncad_jepa' for the fused JEPA+contrastive variant."
        )
    elif canonical == "ncad_jepa":
        model = NCADJEPAModel(
            input_dim=input_dim,
            latent_dim=config.latent_dim,
            filters=config.filters,
            tcn_layers=config.tcn_layers,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
            predictor_hidden_dim=max(64, config.latent_dim * 2),
            predictor_layers=2,
            ema_decay=0.995,
        )
    elif canonical == "ncad_flow_jepa":
        model = NCADFlowJEPAModel(
            input_dim=input_dim,
            latent_dim=config.latent_dim,
            filters=config.filters,
            tcn_layers=config.tcn_layers,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
            predictor_hidden_dim=max(64, config.latent_dim * 2),
            predictor_layers=3,
            ema_decay=0.995,
        )
    elif canonical == "flow_jepa":
        encoder = build_encoder(config, input_dim, device)
        model = FlowTSJEPAModel(
            context_encoder=encoder,
            latent_dim=config.latent_dim,
            predictor_hidden_dim=max(64, config.latent_dim * 2),
            predictor_layers=3,
            ema_decay=0.996,
            dropout=config.dropout,
        )
    elif canonical == "patch_flow_jepa":
        patch_size = config.patch_size
        n_tgt_patches = max(1, config.suspect_size // patch_size)
        model = PatchFlowJEPA(
            input_dim=input_dim,
            patch_size=patch_size,
            d_model=config.filters,
            n_heads=4,
            n_layers=3,
            d_ff=config.filters * 2,
            n_target_patches=n_tgt_patches,
            predictor_layers=3,
            ema_decay=0.996,
            dropout=config.dropout,
        )
    elif canonical == "multiscale_ts_jepa":
        model = MultiScaleTSJEPA(
            input_dim=input_dim,
            latent_dim=config.latent_dim,
            horizons=(config.suspect_size // 2, config.suspect_size),
            filters=config.filters,
            tcn_layers=config.tcn_layers,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
            predictor_hidden_dim=max(64, config.latent_dim * 2),
            ema_decay=0.996,
        )
    elif canonical == "causal_ssm_flow_jepa":
        model = CausalSSMFlowJEPA(
            in_channels=input_dim,
            latent_dim=config.latent_dim,
            hidden_dim=config.filters,
            node_dim=max(16, config.latent_dim),
            ssm_layers=2,
            gat_layers=2,
            num_heads=4,
            flow_layers=3,
            dropout=config.dropout,
            ema_decay=0.996,
        )
    else:
        # Should be unreachable because canonical_model_type raises for unknowns,
        # but keep a defensive error in case a new spec is added without a branch.
        raise ValueError(f"Unsupported model type: {config.model_type!r} (canonical: {canonical!r})")
    return model.to(device)


def limit_windows(windows: np.ndarray, max_windows: Optional[int]) -> np.ndarray:
    """Uniformly sub-sample windows across the full temporal range if max_windows is exceeded."""
    if max_windows is None or len(windows) <= max_windows:
        return windows
    # Uniformly spaced temporal indices covering the full series duration
    indices = np.linspace(0, len(windows) - 1, max_windows, dtype=np.int64)
    return windows[indices]


def encode_windows(model: EncoderModel, windows: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    """Encode an array of temporal sliding windows into latent embeddings."""
    model.eval()
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start : start + batch_size]).float().to(device)
            embeddings.append(model(batch).cpu().numpy())
    if not embeddings:
        return np.empty((0, model.latent_dim), dtype=np.float32)
    return np.concatenate(embeddings, axis=0).astype(np.float32)


def split_train_validation(
    windows: np.ndarray,
    val_split: float = 0.1,
    seed: Optional[int] = None,
    window_size: Optional[int] = None,
    step: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Chronologically split sliding windows into training and validation partitions with a purge gap.

    Purging prevents overlapping time samples between training and validation windows.
    The split is strictly chronological (no shuffling), so *seed* is accepted for
    backward compatibility but ignored.
    """
    del seed  # accepted for backward compatibility; split is chronological
    n_windows = len(windows)
    if n_windows < 10 or val_split <= 0.0:
        return windows, np.empty((0,) + windows.shape[1:], dtype=windows.dtype)

    n_val = max(1, int(n_windows * val_split))
    w_size = window_size if window_size is not None else (windows.shape[1] if windows.ndim >= 2 else 1)
    purge_gap = max(1, int(np.ceil(w_size / max(1, step))))

    # Place validation set at the chronological end
    n_train = n_windows - n_val - purge_gap
    if n_train < 1:
        n_train = max(1, int(n_windows * 0.8))
        val_start = min(n_windows - 1, n_train + 1)
        return windows[:n_train], windows[val_start:]

    train_windows = windows[:n_train]
    val_windows = windows[n_train + purge_gap :]
    return train_windows, val_windows


def train_ts_jepa(
    train_windows: np.ndarray,
    config: CSMConfig,
    input_dim: int,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    """Train TS-JEPA via self-supervised latent dynamics prediction and VICReg loss."""
    model = build_ts_jepa_model(config, input_dim, device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-5)
    injector = ContextualAnomalyInjector(AnomalyInjectionConfig(injection_ratio=config.injection_ratio), seed=config.seed)
    training_data, validation_data = split_train_validation(
        train_windows,
        val_split=config.val_split,
        seed=config.seed,
        window_size=config.full_window_size,
        step=config.step,
    )

    total_steps = config.epochs * max(1, len(training_data) // config.batch_size)
    global_step = 0

    best_state = None
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_indices = np.random.permutation(len(training_data))
        total_loss = 0.0
        total_count = 0

        for batch_start in range(0, len(epoch_indices), config.batch_size):
            global_step += 1
            batch_indices = epoch_indices[batch_start : batch_start + config.batch_size]
            batch_arr = training_data[batch_indices]

            ctx = torch.from_numpy(batch_arr[:, : config.context_size]).float().to(device)
            tgt = torch.from_numpy(batch_arr[:, config.context_size :]).float().to(device)

            loss, _ = model.compute_objective(
                ctx, tgt, config, injector=injector, full_batch=batch_arr
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Cosine EMA schedule from 0.996 to 0.9995
            ema_val = 0.996 + (0.9995 - 0.996) * 0.5 * (1.0 - np.cos(np.pi * global_step / total_steps))
            model.update_target_encoder(decay=ema_val)

            total_loss += float(loss.item()) * len(batch_arr)
            total_count += len(batch_arr)

        scheduler.step()
        train_loss = total_loss / max(total_count, 1)
        val_loss = evaluate_ts_jepa_loss(model, validation_data, config, device)
        if not np.isfinite(val_loss):
            val_loss = train_loss
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        logger.info(f"  epoch {epoch:03d}/{config.epochs}: train_loss={train_loss:.5f}, val_loss={val_loss:.5f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info(f"  early stopping after {epoch} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if config.use_mahalanobis:
        ctx_all = train_windows[:, : config.context_size]
        tgt_all = train_windows[:, config.context_size :]
        model.fit_mahalanobis_covariance(ctx_all, tgt_all)

    history["best_val_loss"] = best_val_loss
    return model, history


def evaluate_ts_jepa_loss(
    model: nn.Module,
    validation_data: np.ndarray,
    config: CSMConfig,
    device: torch.device,
) -> float:
    """Evaluate TS-JEPA loss on validation partition."""
    if len(validation_data) == 0:
        return float("nan")
    model.eval()
    # A deterministic validation injector (fixed seed) keeps validation stable.
    # Non-NCAD models simply ignore it in compute_objective.
    val_injector = ContextualAnomalyInjector(
        AnomalyInjectionConfig(injection_ratio=config.injection_ratio),
        seed=config.seed + 1,
    )
    losses = []
    counts = []
    with torch.no_grad():
        for batch_start in range(0, len(validation_data), config.batch_size):
            batch_arr = validation_data[batch_start : batch_start + config.batch_size]
            ctx = torch.from_numpy(batch_arr[:, : config.context_size]).float().to(device)
            tgt = torch.from_numpy(batch_arr[:, config.context_size :]).float().to(device)

            loss, _ = model.compute_objective(
                ctx, tgt, config,
                injector=val_injector, full_batch=batch_arr,
            )

            losses.append(float(loss.item()) * len(batch_arr))
            counts.append(len(batch_arr))
    return sum(losses) / max(sum(counts), 1)


def train_encoder(
    train_windows: np.ndarray,
    config: CSMConfig,
    input_dim: int,
    device: torch.device,
) -> tuple[EncoderModel, dict]:
    """Train temporal encoder via contextual anomaly injection contrastive learning."""
    model = build_encoder(config, input_dim, device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    injector = ContextualAnomalyInjector(AnomalyInjectionConfig(injection_ratio=config.injection_ratio), seed=config.seed)
    val_injector = ContextualAnomalyInjector(AnomalyInjectionConfig(injection_ratio=config.injection_ratio), seed=config.seed + 1)
    training_data, validation_data = split_train_validation(
        train_windows,
        val_split=config.val_split,
        seed=config.seed,
        window_size=config.full_window_size,
        step=config.step,
    )

    best_state = None
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_indices = np.random.permutation(len(training_data))
        total_loss = 0.0
        total_count = 0
        for batch_start in range(0, len(epoch_indices), config.batch_size):
            batch_indices = epoch_indices[batch_start : batch_start + config.batch_size]
            clean_batch = training_data[batch_indices]
            modified_batch, labels = injector.inject_batch(clean_batch, config.context_size)

            full_tensor = torch.from_numpy(modified_batch).float().to(device)
            context_tensor = torch.from_numpy(clean_batch[:, : config.context_size]).float().to(device)
            label_tensor = torch.from_numpy(labels).float().to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = contrastive_loss(model(full_tensor), model(context_tensor), label_tensor, margin=config.margin)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.item()) * len(clean_batch)
            total_count += len(clean_batch)

        train_loss = total_loss / max(total_count, 1)
        val_loss = evaluate_contrastive_loss(model, validation_data, val_injector, config, device)
        if not np.isfinite(val_loss):
            val_loss = train_loss
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        logger.info(f"  epoch {epoch:03d}/{config.epochs}: train_loss={train_loss:.5f}, val_loss={val_loss:.5f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info(f"  early stopping after {epoch} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_loss"] = best_val_loss
    return model, history


def evaluate_contrastive_loss(
    model: EncoderModel,
    validation_data: np.ndarray,
    injector: ContextualAnomalyInjector,
    config: CSMConfig,
    device: torch.device,
) -> float:
    """Evaluate contrastive loss on validation partition."""
    if len(validation_data) == 0:
        return float("nan")
    model.eval()
    losses = []
    counts = []
    with torch.no_grad():
        for batch_start in range(0, len(validation_data), config.batch_size):
            clean_batch = validation_data[batch_start : batch_start + config.batch_size]
            modified_batch, labels = injector.inject_batch(clean_batch, config.context_size)
            full_tensor = torch.from_numpy(modified_batch).float().to(device)
            context_tensor = torch.from_numpy(clean_batch[:, : config.context_size]).float().to(device)
            label_tensor = torch.from_numpy(labels).float().to(device)
            loss = contrastive_loss(model(full_tensor), model(context_tensor), label_tensor, margin=config.margin)
            losses.append(float(loss.item()) * len(clean_batch))
            counts.append(len(clean_batch))
    return sum(losses) / max(sum(counts), 1)
