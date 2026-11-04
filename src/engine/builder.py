"""Model and encoder builder routines for NCAD-CS."""

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
from src.models.jepa.causal_ssm_flow_jepa import CausalSSMFlowJEPA
from src.models.jepa.dual_timescale_latent_world_jepa import DualTimescaleLatentWorldJEPAModel
from src.models.jepa.flow_ts_jepa import FlowTSJEPAModel
from src.models.jepa.gat_jepa import RelationalGAT_JEPAModel
from src.models.jepa.granger_jepa import GrangerCausalJEPAModel
from src.models.jepa.hamiltonian_jepa import HamiltonianSymplecticJEPAModel
from src.models.jepa.harmonic_spring_jepa import HarmonicSpringJEPAModel
from src.models.jepa.kinematic_adaptive_latent_jepa import KinematicAdaptiveLatentJEPAModel
from src.models.jepa.koopman_jepa import KoopmanJEPAModel
from src.models.jepa.latent_deliberation_jepa import LatentDeliberationJEPAModel
from src.models.jepa.latent_world_jepa import LatentWorldJEPAModel
from src.models.jepa.mdl_jepa import MDLCompressorJEPAModel
from src.models.jepa.memory_bank_latent_world_jepa import MemoryBankLatentWorldJEPAModel
from src.models.jepa.minimalist_jepas import (
    CosineCycleJEPAModel,
    CosineJEPAModel,
    CycleJEPAModel,
)
from src.models.jepa.multiagent_consensus_latent_jepa import MultiAgentConsensusLatentJEPAModel
from src.models.jepa.multiscale_latent_world_jepa import MultiScaleLatentWorldJEPAModel
from src.models.jepa.multiscale_ts_jepa import MultiScaleTSJEPA
from src.models.jepa.ncad_flow_jepa import NCADFlowJEPAModel
from src.models.jepa.ncad_jepa import NCADJEPAModel
from src.models.jepa.operator_entropy_jepa import OperatorEntropyJEPAModel
from src.models.jepa.patch_flow_jepa import PatchFlowJEPA
from src.models.jepa.patch_ts_jepa import PatchTSJEPA
from src.models.jepa.potential_flow_jepa import PotentialFlowJEPAModel
from src.models.jepa.prototype_graph_jepa import PrototypeGraphJEPAModel
from src.models.jepa.recurrent_koopman_jepa import RecurrentKoopmanJEPAModel
from src.models.jepa.reynolds_stress_jepa import ReynoldsStressJEPAModel
from src.models.jepa.selective_latent_world_jepa import SelectiveLatentWorldJEPAModel
from src.models.jepa.spectral_physics_latent_jepa import SpectralPhysicsLatentJEPAModel
from src.models.jepa.tangent_harmonic_jepa import TangentHarmonicJEPAModel
from src.models.jepa.tangent_normal_jepa import TangentNormalJEPAModel
from src.models.jepa.transfer_function_jepa import TransferFunctionJEPAModel
from src.models.jepa.ts_jepa import TSJEPAModel

logger = logging.getLogger("NCAD.engine.builder")

EncoderModel = Union[
    HybridTCNEncoder,
    MultiScaleTCNEncoder,
    RelationalGATEncoder,
    SelectiveSSMContextEncoder,
    TSJEPAModel,
    PatchTSJEPA,
    RelationalGAT_JEPAModel,
    FlowTSJEPAModel,
    PotentialFlowJEPAModel,
    HarmonicSpringJEPAModel,
    CosineJEPAModel,
    CycleJEPAModel,
    CosineCycleJEPAModel,
    KoopmanJEPAModel,
    RecurrentKoopmanJEPAModel,
    TangentNormalJEPAModel,
    PrototypeGraphJEPAModel,
    TransferFunctionJEPAModel,
    MDLCompressorJEPAModel,
    GrangerCausalJEPAModel,
    HamiltonianSymplecticJEPAModel,
    LatentWorldJEPAModel,
    MultiScaleLatentWorldJEPAModel,
    SelectiveLatentWorldJEPAModel,
    MemoryBankLatentWorldJEPAModel,
    DualTimescaleLatentWorldJEPAModel,
    KinematicAdaptiveLatentJEPAModel,
    LatentDeliberationJEPAModel,
    SpectralPhysicsLatentJEPAModel,
    MultiAgentConsensusLatentJEPAModel,
    PatchFlowJEPA,
    MultiScaleTSJEPA,
    NCADJEPAModel,
    NCADFlowJEPAModel,
    CausalSSMFlowJEPA,
    DeltaJEPAModel,
    SpeculativeJEPAModel,
    AlienDualTowerJEPAModel,
    EddyDeltaJEPAModel,
    AnchoredStokesJEPAModel,
    SolitonReynoldsJEPAModel,
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
    elif canonical == "potential_flow_jepa":
        encoder = build_encoder(config, input_dim, device)
        model = PotentialFlowJEPAModel(
            context_encoder=encoder,
            latent_dim=config.latent_dim,
            predictor_hidden_dim=max(64, config.latent_dim * 2),
            predictor_layers=3,
            n_regimes=getattr(config, "n_regimes", 4),
            subspace_dim=getattr(config, "subspace_dim", 8),
            use_regimes=getattr(config, "use_regimes", True),
            ema_decay=0.996,
            dropout=config.dropout,
            codebook_temperature=getattr(config, "grassmann_temperature", 0.1),
            load_balance_weight=getattr(config, "grassmann_load_balance_weight", 0.0),
            eval_hard_routing=getattr(config, "grassmann_eval_hard_routing", True),
        )
    elif canonical == "harmonic_spring_jepa":
        encoder = build_encoder(config, input_dim, device)
        model = HarmonicSpringJEPAModel(
            context_encoder=encoder,
            latent_dim=config.latent_dim,
            hidden_dim=max(64, config.latent_dim * 2),
            rank=getattr(config, "rank", 4),
            eps=1e-3,
            ema_decay=0.996,
            dropout=config.dropout,
        )
    elif canonical == "cosine_jepa":
        encoder = build_encoder(config, input_dim, device)
        model = CosineJEPAModel(
            context_encoder=encoder,
            latent_dim=config.latent_dim,
            hidden_dim=max(64, config.latent_dim * 2),
            predictor_layers=2,
            ema_decay=0.996,
            dropout=config.dropout,
        )
    elif canonical == "cycle_jepa":
        encoder = build_encoder(config, input_dim, device)
        model = CycleJEPAModel(
            context_encoder=encoder,
            latent_dim=config.latent_dim,
            hidden_dim=max(64, config.latent_dim * 2),
            ema_decay=0.996,
            dropout=config.dropout,
        )
    elif canonical == "cosine_cycle_jepa":
        encoder = build_encoder(config, input_dim, device)
        model = CosineCycleJEPAModel(
            context_encoder=encoder,
            latent_dim=config.latent_dim,
            hidden_dim=max(64, config.latent_dim * 2),
            ema_decay=0.996,
            dropout=config.dropout,
        )
    elif canonical == "koopman_jepa":
        encoder = build_encoder(config, input_dim, device)
        model = KoopmanJEPAModel(
            context_encoder=encoder,
            latent_dim=config.latent_dim,
            hidden_dim=max(64, config.latent_dim * 2),
            ema_decay=0.996,
            dropout=config.dropout,
        )
    elif canonical == "recurrent_koopman_jepa":
        model = RecurrentKoopmanJEPAModel(
            input_dim=input_dim,
            latent_dim=config.latent_dim,
            hidden_dim=max(64, config.latent_dim * 2),
            ema_decay=0.996,
            dropout=config.dropout,
        )
    elif canonical == "tangent_normal_jepa":
        encoder = build_encoder(config, input_dim, device)
        model = TangentNormalJEPAModel(
            context_encoder=encoder,
            latent_dim=config.latent_dim,
            subspace_dim=min(4, config.latent_dim // 4),
            hidden_dim=max(64, config.latent_dim * 2),
            ema_decay=0.996,
            dropout=config.dropout,
        )
    elif canonical == "prototype_graph_jepa":
        encoder = build_encoder(config, input_dim, device)
        model = PrototypeGraphJEPAModel(
            context_encoder=encoder,
            latent_dim=config.latent_dim,
            num_prototypes=16,
            temperature=0.5,
            ema_decay=0.996,
            dropout=config.dropout,
        )
    elif canonical == "transfer_function_jepa":
        encoder = build_encoder(config, input_dim, device)
        model = TransferFunctionJEPAModel(
            context_encoder=encoder,
            latent_dim=config.latent_dim,
            hidden_dim=max(64, config.latent_dim * 2),
            ema_decay=0.996,
            dropout=config.dropout,
        )
    elif canonical == "mdl_jepa":
        encoder = build_encoder(config, input_dim, device)
        model = MDLCompressorJEPAModel(
            context_encoder=encoder,
            latent_dim=config.latent_dim,
            hidden_dim=max(64, config.latent_dim * 2),
            ema_decay=0.996,
            dropout=config.dropout,
        )
    elif canonical == "granger_jepa":
        model = GrangerCausalJEPAModel(
            input_dim=input_dim,
            latent_dim=config.latent_dim,
            num_heads=4,
            ema_decay=0.996,
            dropout=config.dropout,
        )
    elif canonical == "hamiltonian_jepa":
        encoder = build_encoder(config, input_dim, device)
        model = HamiltonianSymplecticJEPAModel(
            context_encoder=encoder,
            latent_dim=config.latent_dim,
            hidden_dim=max(64, config.latent_dim * 2),
            step_size=0.1,
            num_leapfrog_steps=3,
            ema_decay=0.996,
            dropout=config.dropout,
        )
    elif canonical == "latent_world_jepa":
        model = LatentWorldJEPAModel(
            input_dim=input_dim,
            latent_dim=config.latent_dim,
            hidden_dim=max(64, config.latent_dim * 2),
            ema_decay=0.996,
            dropout=config.dropout,
        )
    elif canonical == "multiscale_latent_world_jepa":
        model = MultiScaleLatentWorldJEPAModel(
            input_dim=input_dim,
            latent_dim=config.latent_dim,
            hidden_dim=max(64, config.latent_dim * 2),
            ema_decay=0.996,
            dropout=config.dropout,
        )
    elif canonical == "selective_latent_world_jepa":
        model = SelectiveLatentWorldJEPAModel(
            input_dim=input_dim,
            latent_dim=config.latent_dim,
            d_state=8,
            hidden_dim=max(64, config.latent_dim * 2),
            ema_decay=0.996,
            dropout=config.dropout,
        )
    elif canonical == "memory_bank_latent_world_jepa":
        model = MemoryBankLatentWorldJEPAModel(
            input_dim=input_dim,
            latent_dim=config.latent_dim,
            num_heads=4,
            hidden_dim=max(64, config.latent_dim * 2),
            ema_decay=0.996,
            dropout=config.dropout,
        )
    elif canonical == "dual_timescale_latent_world_jepa":
        model = DualTimescaleLatentWorldJEPAModel(
            input_dim=input_dim,
            latent_dim=config.latent_dim,
            k_interval=8,
            hidden_dim=max(64, config.latent_dim * 2),
            ema_decay=0.996,
            dropout=config.dropout,
        )
    elif canonical == "kinematic_adaptive_latent_jepa":
        model = KinematicAdaptiveLatentJEPAModel(
            input_dim=input_dim,
            latent_dim=config.latent_dim,
            hidden_dim=max(64, config.latent_dim * 2),
            ema_decay=0.996,
            dropout=config.dropout,
        )
    elif canonical == "latent_deliberation_jepa":
        model = LatentDeliberationJEPAModel(
            input_dim=input_dim,
            latent_dim=config.latent_dim,
            hidden_dim=max(64, config.latent_dim * 2),
            num_thought_steps=4,
            num_heads=4,
            ema_decay=0.996,
            dropout=config.dropout,
        )
    elif canonical == "spectral_physics_latent_jepa":
        model = SpectralPhysicsLatentJEPAModel(
            input_dim=input_dim,
            latent_dim=config.latent_dim,
            hidden_dim=max(64, config.latent_dim * 2),
            ema_decay=0.996,
            dropout=config.dropout,
        )
    elif canonical == "multiagent_consensus_latent_jepa":
        model = MultiAgentConsensusLatentJEPAModel(
            input_dim=input_dim,
            d_agent=16,
            hidden_dim=max(32, config.latent_dim),
            num_heads=2,
            ema_decay=0.996,
            dropout=config.dropout,
        )
    elif canonical == "tangent_harmonic_jepa":
        encoder = build_encoder(config, input_dim, device)
        model = TangentHarmonicJEPAModel(
            context_encoder=encoder,
            latent_dim=config.latent_dim,
            hidden_dim=max(64, config.latent_dim * 2),
            predictor_layers=2,
            ema_decay=0.996,
            alpha=getattr(config, "harmonic_alpha", 1.0),
            beta=getattr(config, "harmonic_beta", 0.5),
            dropout=config.dropout,
        )
    elif canonical == "reynolds_stress_jepa":
        encoder = build_encoder(config, input_dim, device)
        model = ReynoldsStressJEPAModel(
            context_encoder=encoder,
            latent_dim=config.latent_dim,
            stress_dim=min(16, config.latent_dim),
            hidden_dim=max(64, config.latent_dim * 2),
            predictor_layers=2,
            ema_decay=0.996,
            alpha_stress=0.5,
            alpha_cone=0.2,
            dropout=config.dropout,
        )
    elif canonical == "operator_entropy_jepa":
        encoder = build_encoder(config, input_dim, device)
        model = OperatorEntropyJEPAModel(
            context_encoder=encoder,
            latent_dim=config.latent_dim,
            hidden_dim=max(64, config.latent_dim * 2),
            predictor_layers=2,
            ema_decay=0.996,
            alpha_entropy=getattr(config, "alpha_entropy", 0.5),
            alpha_rep_entropy=getattr(config, "alpha_rep_entropy", 0.0),
            beta_lyapunov=getattr(config, "beta_lyapunov", 0.02),
            spectral_gap_min=0.1,
            dropout=config.dropout,
            use_radial_heat=getattr(config, "use_radial_heat", False),
            use_cohn_elkies=getattr(config, "use_cohn_elkies", False),
            use_koopman_matrix=getattr(config, "use_koopman_matrix", True),
            use_cayley_operator=getattr(config, "use_cayley_operator", False),
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
    else:
        raise ValueError(f"Unsupported model type: {config.model_type!r} (canonical: {canonical!r})")

    return model.to(device)
