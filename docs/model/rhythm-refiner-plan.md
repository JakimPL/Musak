# Rhythm Refiner Plan

This document defines the isolated rhythm-refiner experiment. It narrows the broader rhythm and figure planning ideas
from [rhythm-figure-plan.md](rhythm-figure-plan.md) into a small model that can be trained, evaluated, and discarded
without changing the main token decoder.

The refiner should answer one question first:

```text
Can a model learn whole-span rhythm and hand-communication structure better than token-by-token generation?
```

If it cannot answer that in isolation, it should not be wired into the main model.

## Scope

The first refiner is a rhythm-grid model, not a full composition model.

It predicts or scores:

- per-hand grid-cell activity;
- coactivity between hands;
- missing-hand completions;
- continuation compatibility;
- real-vs-corrupted rhythm grids.

It does not predict:

- exact pitch;
- harmonic spelling;
- final token streams;
- recursive split trees;
- figure embeddings.

Those can come later only if the rhythm-grid task proves useful.

## Representation

Each segment is converted into a fixed grid over the requested bar span.

The default training grid is sixteenth-note cells. This is still coarse enough for the first classifier, but it covers
common pickup and short-bar spans such as `5/16` that cannot be represented on an eighth-note grid. Samples that still
cannot be represented on the configured grid are skipped and counted during frame construction.

Per hand, every cell has one state:

```text
unknown
rest
onset
sustain
```

`unknown` is a model-input state used for masking. It is not a target emitted by the extractor. `rest` is a committed
musical target.

The first extraction target uses:

```text
rest
onset
sustain
```

The grid is conditioned by:

- meter position;
- bar index and distance to end;
- hand;
- harmonic plan fields when available;
- phrase or cadence role when available;
- optional difficulty or density controls.

The first implementation should use exact rational positions or integer ticks. Notebook data frames may use floats.

## Why This Is Worth Trying

The main decoder sees a prefix and predicts the next token. It has no separate object for "the next bar answers the
previous bar" or "the left hand should support this right-hand rhythm". A grid refiner can be trained on whole spans:

```text
partially known rhythm grid -> completed rhythm grid
```

This makes hand communication and continuation testable before pitch generation enters the picture.

## Critical Risks

1. The model may learn density rather than phrasing.
2. One-bar windows may be too short for recurrence; two-bar windows may be sparse.
3. Good rhythm does not guarantee good music without harmony, register, and pitch filling.
4. Reconstruction loss can improve while generation quality stays unchanged.
5. Corruption tasks can be too easy unless negatives are matched by length, meter, hand activity, and density.

These risks define the evaluation. The refiner must beat simple baselines on controlled tasks before integration.

## Phase 0: Grid Extraction

Goal: create the stable data representation.

Status: implemented for `Segment -> RhythmGridFrame` extraction and basic grid metrics.

Implementation tasks:

- Convert `Segment` objects to exact rhythm grids.
- Emit one target state per hand and grid cell.
- Distinguish note attacks from sustained notes.
- Treat chords as one onset in the hand activity grid.
- Respect pickup and short-final bar durations from segment metadata.
- Derive deterministic coactivity labels from per-hand states.
- Add unit tests before training code.

Acceptance criteria:

- A sustained whole-bar note produces one `onset` cell followed by `sustain` cells.
- A rest-only hand produces `rest` cells.
- Two simultaneous hand onsets produce `both_synchronized`.
- Short bars produce fewer cells and exact coverage.

## Phase 1: Descriptive Metrics

Goal: measure current generated rhythm failures without training the refiner.

Metrics:

- per-hand onset rate;
- per-hand sustain rate;
- whole-bar sustain without answer rate;
- coactivity distribution;
- one-hand-only activity rate;
- both-hand synchronized onset rate;
- rest/onset/sustain entropy;
- bar-level activity-template recurrence;
- recurrence without stasis.

These metrics should be logged for corpus and generation before they become losses or rewards.

## Phase 2: Masked Classifier

Goal: train the first isolated model.

Status: initial `make train-refiner` path implemented for masked per-cell activity and coactivity classification.

Input:

```text
masked grid states
meter coordinates
hand IDs
bar/distance-to-end IDs
optional harmony IDs
```

Targets:

```text
per-cell hand activity
coactivity
next-onset distance bucket
```

Recommended first architecture:

```text
state embeddings
+ coordinate embeddings
+ optional harmony embeddings
-> small Transformer or TCN
-> activity and coactivity heads
```

This is still a classifier. It is not the main decoder.

## Phase 3: Masking Tasks

Training should mix several masks:

- random cells hidden;
- whole bar hidden;
- one hand hidden;
- ending hidden;
- interior hidden with boundary anchors visible;
- phrase continuation hidden.

The most important early task is one-hand-hidden reconstruction, because it directly tests whether the model learns
communication between hands.

## Phase 4: Compatibility Tasks

Add ranking or binary classification tasks:

- true continuation vs shuffled continuation;
- true left/right pair vs mismatched pair;
- real phrase vs whole-bar-stasis corruption;
- real phrase vs density-matched random grid;
- true varied recurrence vs unrelated phrase.

Density-matched negatives are mandatory. Otherwise the model can win by learning surface activity counts.

## Phase 5: Standalone Evaluation

Do not wire into generation until standalone evaluation is convincing.

Metrics:

- onset precision, recall, and F1;
- rest and sustain F1;
- one-hand-hidden reconstruction accuracy;
- ending reconstruction accuracy;
- coactivity accuracy;
- real-vs-corrupt AUC;
- continuation ranking accuracy;
- left/right compatibility ranking accuracy;
- whole-bar stasis false-positive rate.

Baselines:

- independent per-cell classifier;
- copy previous bar;
- sample corpus rhythm template;
- rhythm n-gram baseline.

## Phase 6: Iterative Refinement

Only after the masked classifier works, test iterative refinement:

```text
start from mostly unknown grid
predict high-confidence cells
commit selected cells
mask uncertain regions
repeat until stable or shortest pulse reached
```

Do not train explicit split/stop decisions first. Iterative masked refinement gives a similar coarse-to-fine behavior
with fewer procedural assumptions.

## Phase 7: Main-Model Integration

Integration is justified only if the isolated refiner beats baselines and improves manual grid inspection.

Possible integration paths:

1. Generate a rhythm grid first, then constrain or bias token decoding.
2. Add rhythm-grid conditioning to the decoder.
3. Use the refiner as a candidate scorer or reranker.
4. Use learned phrase embeddings for retrieval before token decoding.

The first integration should be reversible through a config flag.

## Commands

Planned commands:

```bash
make train-refiner
make evaluate-refiner
```

These should delegate to a shared refiner module, use YAML configs, and log separate MLflow runs. They should not reuse
the main pretraining command with many CLI overrides.

## Decision Gates

Stop or simplify if:

- grid metrics cannot distinguish corpus from known bad generations;
- one-hand-hidden reconstruction is no better than simple baselines;
- continuation ranking fails against density-matched negatives;
- generated rhythm grids collapse into stasis or mechanical repetition;
- wiring the refiner improves plan-following metrics but not musical inspection.

The refiner is useful only if it learns communication and continuation, not just legal rhythm density.
