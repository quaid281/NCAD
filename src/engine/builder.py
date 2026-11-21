"""Model and encoder builder routines for Laminar-JEPA-Lite / NCAD-CS."""

from __future__ import annotations

import logging
from typing import Union

import torch
import torch.nn as nn

from src.config import CSMConfig
from src.models.encoders.multi_scale_tcn_encoder import MultiScaleTCNEncoder
from src.models.encoders.relational_gat_encoder import RelationalGATEncoder
from src.models.encoders.selective_ssm_encoder import SelectiveSSMContextEncoder
from src.models.encoders.tcn_encoder import HybridTCNEncoder
from src.models.jepa.alien_jepas import (
    AlienDualTowerJEPAModel,
    AnchoredStokesJEPAModel,
    DeltaJEPAModel,
    EddyDeltaJEPAModel,
    HelmholtzSolitonJEPAModel,
    SolitonReynoldsJEPAModel,
    SpeculativeJEPAModel,
)
from src.models.jepa.multiscale_ts_jepa import MultiScaleTSJEPA
from src.models.jepa.ts_jepa import TSJEPAModel
from src.models.laminar_lite import LaminarJEPALiteModel, LaminarLiteEncoder

logger = logging.getLogger("NCAD.engine.builder")

EncoderModel = Union[
    HybridTCNEncoder,
    MultiScaleTCNEncoder,
    RelationalGATEncoder,
    SelectiveSSMContextEncoder,
    TSJEPAModel,
    MultiScaleTSJEPA,
    DeltaJEPAModel,
    SpeculativeJEPAModel,
    AlienDualTowerJEPAModel,
    EddyDeltaJEPAModel,
    AnchoredStokesJEPAModel,
    SolitonReynoldsJEPAModel,
    HelmholtzSolitonJEPAModel,
    LaminarJEPALiteModel,
]


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
        model = RelationalGATEncoder(**common_kwargs)
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
    """Instantiate and initialize the requested JEPA model variant."""
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
    elif canonical == "ncad":
        raise ValueError(
            "The 'ncad' model type uses the contrastive encoder path, not build_ts_jepa_model."
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
    elif canonical == "delta_jepa":
        encoder = build_encoder(config, input_dim, device)
        model = DeltaJEPAModel(
            context_encoder=encoder,
            latent_dim=config.latent_dim,
            predictor_hidden_dim=max(64, config.latent_dim * 2),
            ema_decay=0.996,
            dropout=config.dropout,
        )
    elif canonical == "speculative_jepa":
        encoder = build_encoder(config, input_dim, device)
        model = SpeculativeJEPAModel(
            context_encoder=encoder,
            latent_dim=config.latent_dim,
            predictor_hidden_dim=max(64, config.latent_dim * 2),
            ema_decay=0.996,
            dropout=config.dropout,
        )
    elif canonical == "alien_dual_tower_jepa":
        encoder = build_encoder(config, input_dim, device)
        model = AlienDualTowerJEPAModel(
            context_encoder=encoder,
            latent_dim=config.latent_dim,
            predictor_hidden_dim=max(64, config.latent_dim * 2),
            ema_decay=0.996,
            dropout=config.dropout,
        )
    elif canonical == "eddy_delta_jepa":
        encoder = build_encoder(config, input_dim, device)
        model = EddyDeltaJEPAModel(
            context_encoder=encoder,
            latent_dim=config.latent_dim,
            predictor_hidden_dim=max(64, config.latent_dim * 2),
            ema_decay=0.996,
            dropout=config.dropout,
        )
    elif canonical == "anchored_stokes_jepa":
        encoder = build_encoder(config, input_dim, device)
        model = AnchoredStokesJEPAModel(
            context_encoder=encoder,
            latent_dim=config.latent_dim,
            predictor_hidden_dim=max(64, config.latent_dim * 2),
            ema_decay=0.996,
            dropout=config.dropout,
        )
    elif canonical == "soliton_reynolds_jepa":
        encoder = build_encoder(config, input_dim, device)
        model = SolitonReynoldsJEPAModel(
            context_encoder=encoder,
            latent_dim=config.latent_dim,
            predictor_hidden_dim=max(64, config.latent_dim * 2),
            ema_decay=0.996,
            dropout=config.dropout,
        )
    elif canonical == "helmholtz_soliton_jepa":
        encoder = build_encoder(config, input_dim, device)
        model = HelmholtzSolitonJEPAModel(
            context_encoder=encoder,
            latent_dim=config.latent_dim,
            predictor_hidden_dim=max(64, config.latent_dim * 2),
            ema_decay=0.996,
            dropout=config.dropout,
        )
    elif canonical in ("laminar_jepa_lite", "laminar_jepa_224"):
        num_horizons = 4 if canonical == "laminar_jepa_224" else 3
        lite_encoder = LaminarLiteEncoder(
            input_dim=input_dim,
            latent_dim=config.latent_dim,
            filters=config.filters,
            tcn_layers=min(4, config.tcn_layers),
            kernel_size=config.kernel_size,
        )
        model = LaminarJEPALiteModel(
            context_encoder=lite_encoder,
            latent_dim=config.latent_dim,
            predictor_hidden_dim=max(64, config.latent_dim * 2),
            ema_decay=0.996,
            num_horizons=num_horizons,
        )
    else:
        raise ValueError(f"Unsupported model type: {config.model_type!r} (canonical: {canonical!r})")

    return model.to(device)
