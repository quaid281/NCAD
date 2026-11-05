---
name: loss-and-architecture-parsimony
description: Enforce strict mathematical parsimony in neural network loss functions and geometric architectures. Use this skill whenever designing, modifying, auditing, or reviewing model forward passes, geometric layers, or compute_objective / training loss functions in src/models/.
---

# Loss Function & Architectural Parsimony Skill

Whenever designing, editing, or auditing models in `src/models/` (especially `ReynoldsStressJEPAModel` and geometric layers in `src/models/geometric_layers.py`), enforce the following rules without exception:

## 1. Architectural Invariants Over Loss Penalties
- **Never add auxiliary regularizer soup to `compute_objective`** when an invariant can be parameterized structurally inside a layer:
  - **Contractivity & Numerical Range:** Enforce via Cayley-Crouzeix parameterization $\mathbf{W} = (\mathbf{I} + \mathbf{S} + \mathbf{D}_{\text{visc}})^{-1}(\mathbf{I} - \mathbf{S} - \mathbf{D}_{\text{visc}})$ (`CrouzeixDissipativePredictor`), never via per-batch SVD penalties in the loss.
  - **Orthogonality:** Enforce via QR retraction on $\mathrm{St}(d_s, D)$ (`StiefelLinear`), never via $\|\mathbf{P}\mathbf{P}^\top - \mathbf{I}\|_F^2$ loss penalties.
  - **Positive-Definite Stress Cone ($\mathcal{S}_+^{d_s}$):** Enforce via Stiefel shear modes with positive weights $\sum_k c_k(z) v_k v_k^\top + \gamma \mathbf{I}$ (`AdmissibleStressConeProjection`), never via eigenvalue hinge penalties (`cone_defect == 0.0` identically).
  - **Fourier Positivity & Moment Bounds:** Enforce via positive normalized quadrature weights (`CohnElkiesFilter`) and similarity-invariant `eigvalsh` resolvent traces (`HankelMomentFilter`), never via hard `torch.clamp` cutoffs.

## 2. Two-Term Objective Parsimony (`ReynoldsStressJEPAModel`)
- The training objective of `ReynoldsStressJEPAModel.compute_objective` must consist exclusively of the two physical Reynolds decomposition alignment terms:
  $$\mathcal{L}_{\text{Reynolds}} = \underbrace{\|z_{\text{pred}} - \operatorname{sg}(z_{\text{tgt}})\|_2^2}_{\text{1st-Order Mean-Flow Alignment}} + \alpha_{\text{stress}} \underbrace{\|\boldsymbol{\Sigma}_{\text{pred}} - \operatorname{sg}(\boldsymbol{\Sigma}_{\text{obs}})\|_F^2}_{\text{2nd-Order Stress Closure Alignment}}$$
- **Prohibited in `compute_objective`:**
  1. Importing loss functions from other JEPA variants (e.g., `spectral_operator_entropy_loss` or `compute_empirical_covariance`).
  2. Running duplicate forward passes (e.g., calling `self._predict(z_ctx.detach())` a second time).
  3. Adding trace bonuses or zero-gradient terms on stop-gradient targets (e.g., `rayleigh_capture`).

## 3. Zero Redundant Clamps & Zero Duplicate Submodule Calls
- Every leaf submodule must execute **at most once** per forward pass (`hankel_filter` on context only, `cohn_elkies` on predictor only, `stress_proj` once).
- Never stack a heuristic `torch.tanh` or `torch.clamp` on top of an operator that is already algebraically bounded (such as resolvent eigenvalues $\lambda / (|\lambda| + \gamma) \in (-1, 1)$ or non-positive Mellin-Fourier envelopes $h_\epsilon(\zeta) \in [-2, 0]$).
- Maintain **degree-1 scale homogeneity** ($D(c \cdot e) = c \cdot D(e)$ for all $c > 0$) in `CoordinateSaliencyGate` by normalizing coordinate magnitudes by the coordinate RMS before softmax weighting.

## 4. Zero Layer, Depth, and Structural Redundancy (`[INV-NO-REDUNDANCY]`)
- **No Back-to-Back / Algebraically Redundant Layers:** Never stack consecutive normalizations (`nn.LayerNorm` followed immediately by `F.layer_norm` without an intervening nonlinear transformation), consecutive linear projections without a nonlinearity ($\mathbf{W}_2 \mathbf{W}_1 \equiv \mathbf{W}$), or dead identity branches.
- **Activation-Trace Functional Necessity:** Every depth block or projection head must justify its existence in `LayerActivationTracer` by preserving or increasing effective rank ($R_{\text{stab}}$) or anomaly Fisher separation. Layers that attenuate anomaly divergence without improving representation conditioning must be bypassed via multi-depth taps or pruned.
- **Canonical Operator Reuse (Zero Copy-Paste Boilerplate):** Shared multi-horizon whitening, envelope scoring, covariance fitting, and fast-weight rollouts must be factored into single canonical primitives rather than duplicated across model classes. Never invoke `self.forward(c, t)` twice in a lambda (`f(x)[1] - f(x)[2]`) when a single forward pass suffices.
