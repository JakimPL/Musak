# Musak Model Improvement Roadmap

This document turns the research direction from
[symbolic-music-literature-review.md](symbolic-music-literature-review.md) into a staged engineering plan for
`musak_model`.

Before implementing any phase, follow [guidelines.md](guidelines.md) for code ownership, typing, tests, and conversion
placement. For the stochastic generator context and the reference-prior design, use [generator.md](generator.md) as the
main design source. For the current model pipeline, token semantics, and training flow, use [model.md](model.md).

## Current Diagnosis

The current neural model is trained as next-token prediction over a flattened token vocabulary. That makes it good at
local validity and common token-kind patterns, but weak at phrase structure, motivic variation, harmonic intent, and
long-range register/rhythm planning.

The main causes are:

- **Flattened token cross-product.** Degree, accidental, octave, and duration are packed into one token ID, so the
  model must learn attribute structure from data.
- **Weak structural path.** The code supports CNN/GRU hierarchy, but default configs currently disable both, leaving a
  plain causal Transformer path plus conditioning prefix.
- **Objective mismatch.** Cross-entropy plus validity penalty teaches legal next tokens, not musical plans.
- **Underused reference data.** Figure artifacts preserve useful local idioms and rich occurrence context, but
  generation currently uses them mainly in the stochastic generator and evaluation, not as a learned temporal prior.
- **Sparse exact figures.** Exact n-grams are too many to become model tokens. They should be reference priors,
  emissions, and metrics.

## Design Direction

Use a two-layer generative model:

```text
planner / controller
  -> phrase role, density, register curve, hand activity, chord track, figure family, base duration
  -> empirical figure sampler with backoff reference priors
  -> deterministic token renderer + hard playability constraints
```

The planner should operate on low-cardinality musical objects. Exact figure templates should stay in a reference index
and renderer. This keeps the model controllable, data-efficient, and compatible with Musak's current scale-degree token
semantics.

## Priority Roadmap

### 0. Literature Review And Design Synthesis

Deliverable: [symbolic-music-literature-review.md](symbolic-music-literature-review.md).

Acceptance criteria:

- Covers symbolic representation, hierarchy, controllability, constraints, and evaluation.
- Lists actionable implications for Musak.
- Links all cited papers or project pages.

### 1. Factorize The Neural Token Objective

Goal: reduce the burden of the 1611-way vocabulary and expose musical attributes to the model.

Implementation direction:

- Add a factorized prediction path that predicts token kind first.
- For `NoteToken`, predict degree, accidental, octave offset, and duration as separate heads.
- For `RestToken` and `HoldToken`, predict duration through the duration head.
- Keep the existing flat vocabulary path available until the factorized path is validated.
- Compare flat loss, factorized loss, token-kind accuracy, attribute accuracy, validity, and generation quality.

Acceptance criteria:

- Teacher-forced validation loss is comparable or better than the flat baseline.
- Attribute-level metrics expose whether failures are pitch, duration, register, or token-kind errors.
- Generated samples have no regression in hard-constraint validity.

### 2. Build A Figure-Family Reference Index

Goal: turn n-grams into a usable prior without making them tokens.

Implementation direction:

- Define a low-cardinality `FigureFamily` from existing figure properties:
  contour shape, duration shape, monophonic/chordal flags, in-scale flag, opening interval class, closing interval
  class, and figure length.
- Add transition counts over ordered figure occurrences per sample and hand.
- Add conditional counts keyed by family, hand, scale type, metrical position, anchor bin, base duration, and optional
  chord context.
- Export these as new reference artifacts while preserving existing `figure/all/counts.parquet` semantics.

Acceptance criteria:

- Existing figure-profile metrics remain unchanged.
- New artifacts can answer:
  `P(family_t | family_{t-1}, hand, meter position, chord context)` and
  `P(figure | family, scale_type, hand, n, anchor/bin, meter position)`.
- Backoff behavior is deterministic and tested.

### 3. Close The Stochastic Generator Fitting Loop

Goal: make the non-neural generator a calibrated musical baseline instead of a hand-set process.

Implementation direction:

- Fit register-curve parameters per `(scale_type, hand)` from onset register trajectories.
- Fit accent/density/hand-activity parameters from diagnostics and rhythm artifacts.
- Use decoded chord tracks to build empirical chord transition models.
- Use figure-family and exact-figure conditionals in substitution scoring.
- Keep hard playability in `GenerationConstraintState`; stochastic components stay soft.

Acceptance criteria:

- Synthetic output improves against reference metrics without requiring manual slider tuning.
- Register autocorrelation, density, rhythm, harmonic consonance, and figure-family distributions are logged.
- `generator.md` and `generator-model.md` are updated when code behavior changes.

### 4. Establish The Stochastic Generator As Baseline

Goal: create a reliable baseline for musicality, controllability, and synthetic data generation.

Implementation direction:

- Add a reproducible calibration command for selected meters/scales/difficulty bands.
- Store chosen generator configs as versioned artifacts.
- Compare stochastic output, neural output, and training reference on the same metrics.
- Use the baseline to generate synthetic curricula only after reference-distance thresholds are met.

Acceptance criteria:

- Baseline reports are reproducible from one command.
- Generated samples satisfy hard constraints and complete requested bars.
- Synthetic data is not mixed into neural training unless it passes documented metric thresholds.

### 5. Train A Neural Planner

Goal: learn musical structure at the right abstraction level.

Implementation direction:

- Build planner training examples from processed segments and figure-family artifacts.
- Inputs: scale type, meter, structural controls, difficulty when reliable, previous families, previous planned state.
- Targets: next phrase role, chord/function, register bin, density/activity, figure family, and base duration.
- Start with a small GRU/Transformer over bar or cell states.
- Decode plans through the empirical figure sampler and token renderer.

Acceptance criteria:

- Planner generations beat the stochastic baseline on coherence metrics while staying near reference distributions.
- Controls reliably affect generated output.
- Exact figure novelty is allowed, but family and rhythm distributions remain reference-like.

### 6. Add A Plan-Conditioned Token Decoder Only If Needed

Goal: use raw-token modeling as refinement, not as the primary source of structure.

Implementation direction:

- Feed planner states as per-bar or per-cell memory, not only as one prefix vector.
- Add shallow-fusion decoding with reference priors:

```text
logits' = logits
        + alpha * log P_reference(next_token | figure_prefix, context)
        + beta  * validity/reference penalties
```

- Keep renderer-only generation as the baseline and compare against the learned decoder.

Acceptance criteria:

- The token decoder improves surface quality without weakening plan adherence.
- It does not reintroduce repetition collapse or incoherent structure.
- It is optional in production generation paths.

## Evaluation Plan

Use the same metric families for stochastic, planner, and token-decoder outputs:

- hard validity: completed bars, constraint failures, decode errors;
- reference alignment: figure TV distance, common/rare/novel figure mass, property/contour/duration-shape TV;
- rhythm: onset density, strong-beat distribution, IOI/duration distributions, grid alignment;
- pitch/register: range, static hand span, melodic gaps, register lag-1 autocorrelation;
- harmony: chord-tone fit by metrical strength, coincident-onset consonance, decoded chord-transition plausibility;
- structure: repeated figure-family rate, variation-after-repeat rate, phrase-begin/end stability, excessive repetition;
- control accuracy: requested meter, scale, density, range, duration floor, chordal/monophonic texture, difficulty band.

Human inspection remains required for research milestones. A sample should be inspected as notation, piano roll, and
audio before treating a metric improvement as real.

## Implementation Defaults

- Do not add exact n-grams as model tokens.
- Preserve scale-relative pitch semantics and `scale_root` as decode metadata.
- Add abstractions only where they serve a measurable phase goal.
- Keep generated-artifact schemas versioned and fail fast on partial/incompatible artifact sets.
- Keep docs synchronized: update this roadmap for high-level plan changes, [generator.md](generator.md) for generator
  design changes, [generator-model.md](generator-model.md) for implemented generator behavior, and [model.md](model.md)
  for token/training pipeline changes.
