# Isolated Novelty Experiments

These files are research branches for NCAD-CS v3. They do not change the default `train.py` pipeline and are not imported by the production notebook unless we explicitly do so.

## Idea A: Hopfield Context Memory

`hopfield_context.py` replaces K-Means prototypes with a continuous Hopfield-style associative retrieval mechanism. The contamination distance is the residual between an observed context embedding and its retrieved normal attractor state.

Reviewer framing: this is not clustering-based anomaly detection. It is energy-based associative context retrieval used only to support counterfactual substitution.

## Idea B: Counterfactual Context Substitution

`causal_counterfactual.py` learns a normal-only structural equation from context embeddings to full-window embeddings. At inference, a suspicious context can be replaced by an intervened healthy context and rescored as `do(context = healthy_context)`.

Reviewer framing: the core method becomes causal intervention over context, not nearest-prototype scoring.

## Idea D: Selective SSM Context Encoder

`selective_ssm_encoder.py` provides a Mamba-inspired encoder with input-dependent state updates. It has the same input and output shape as the TCN encoder, so it can be tested as an isolated encoder branch.

Reviewer framing: contaminated context is treated as a selective state update problem, where the model can learn which temporal inputs should update or be suppressed in the latent state.

## No Memory Bank Variant

`memoryless_context.py` implements the contaminate-vs-normal idea without a memory bank. It trains a compact neural head that classifies context contamination and denoises contaminated context embeddings back toward the normal context manifold. At inference it uses only learned weights and calibration thresholds, with no stored prototypes or nearest-neighbor lookup.

Reviewer framing: this is a prototype-free context intervention model. The method can be described as self-supervised contamination discrimination plus learned context denoising.

## Smoke Test

From the repository root:

```bash
python notebooks_v3/experimental/run_experimental_smoke.py
```