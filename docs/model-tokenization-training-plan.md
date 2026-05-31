# Model Tokenization And Training Improvement Plan

This document is the concrete implementation plan for the first improvement cycle after
[model-improvement-roadmap.md](model-improvement-roadmap.md). It focuses on the issues that currently block efficient
learning and useful generation:

- scale-family, modal, and chromatic material can be encoded with too little orthographic context;
- the neural model predicts one flattened token ID instead of modeling the musical attributes that make up that token;
- the training objective mostly rewards legal next-token reconstruction, while sight-reading exercises require
  controlled rhythm, register, density, voice leading, and harmonic plausibility.

Follow [guidelines.md](guidelines.md) for ownership, typing, tests, and conversion placement before implementing any
phase. Update [model.md](model.md) whenever token semantics, encoded artifact schemas, training inputs, or generation
constraints change.

## Current Problem

The current scale matcher chooses a pitch-set basis from duration-weighted pitch-class support. This is useful for
filtering noisy material, but it is not tonal analysis. Natural minor and modes may intentionally collapse to a related
major pitch set. That is acceptable for the current token basis, but it means the model-facing scale coordinates do not
carry tonic function or mode by themselves. Spelling and accidentals therefore need explicit orthographic context rather
than relying on pitch-set root alone.

The current vocabulary also packs note degree, accidental, octave offset, and duration into one ID. With the default
duration vocabulary this creates a 1611-way prediction problem, even though most of the structure is low-cardinality
and shared across token kinds. The model has to rediscover that structure from a small dataset.

Finally, validation loss and token perplexity are not sufficient musical objectives. They can show whether the model
fits teacher-forced tokens, but they do not say whether generated samples stay in the exercise domain, avoid degenerate
loops, preserve a playable texture, or follow a plausible harmonic/rhythmic contour. Full training should wait until
the model logs objective terms that are close to the musical constraints we actually care about.

## Research-Informed Objective Direction

Recent symbolic-music systems usually keep maximum-likelihood or denoising losses as a base learning signal, then add
music-specific structure elsewhere:

- Music Transformer improved long-sequence generation through relative attention for musical timing and repetition,
  not by treating perplexity as the whole quality metric.
- REMI and Pop Music Transformer improved results by changing the event representation to expose bar, beat, rhythm,
  chord, and tempo context to the model.
- Compound Word Transformer groups heterogeneous musical attributes into compound tokens and predicts token types with
  separate heads, matching the direction of a factorized Musak objective.
- MusicBERT uses OctupleMIDI and bar-level masking for symbolic representation learning, showing that bar-aware
  masking/pretraining is more appropriate than blindly porting NLP masking.
- FIGARO and MuseCoco extract high-level musical attributes from the target sequence and train controllable
  description-to-sequence or attribute-to-music models. This is a good fit for short exercises because density,
  register, hand activity, rhythmic shape, and difficulty can be extracted without human labels.
- Rule-guided diffusion and older ML-plus-RL music work use rule or reward functions to steer generated samples, but
  these are better treated as later-stage guidance or reranking until we have a strong supervised baseline.

For Musak, the practical conclusion is: keep cross-entropy for reconstructing valid symbolic events, but do not make it
the only objective. Add supervised musical attribute targets that can be extracted from data, and use reference
distributions as diagnostics, decoding priors, or rerankers before attempting adversarial, RL, or reward-model training.

## Recommended Objective Stack

The first trainable objective should be a multi-term supervised objective, not a single perplexity target:

```text
loss = event_loss
     + auxiliary_weight * musical_attribute_loss
     + grammar_weight * context_validity_loss
```

Where:

- `event_loss` is the masked factorized cross-entropy from Phase 3.
- `musical_attribute_loss` is the sum of supervised sequence-level and bar-level targets from Phase 5.
- `context_validity_loss` is optional and should only penalize probability mass assigned to impossible continuations
  under the current grammar and teacher-forced prefix. It is not a proxy for musicality.

Generated-sample priors should initially live outside the gradient path:

```text
sample_score = hard_validity_score
             + rhythm_reference_score
             + density_reference_score
             + register_reference_score
             + figure_family_reference_score
             + harmonic_reference_score
```

Use this score for evaluation, reranking, and shallow fusion before using it as a reinforcement-learning reward. This
keeps the model honest: if a term improves samples, we can see which musical property improved.

## Design Principles

1. Preserve pitch-set-relative token semantics, and keep spelling or modal context explicit instead of pretending the
   pitch-set root is a tonal tonic.
2. Separate pitch-set confidence, orthographic spelling hints, and musical objective signals.
3. Keep exact figure n-grams out of the neural token vocabulary.
4. Add factorized and musical auxiliary objectives before adding larger model capacity.
5. Treat legality as a hard generation constraint and auxiliary metric, not as the main musicality objective.
6. Make every phase measurable against the current flat-token baseline and against generated-sample metrics.
7. Treat tokenizer schema changes as artifact-version changes. Rebuild processed datasets rather than preserving stale
   encoded artifacts.

## Progress Log

This section is the durable resumption point if work continues after context compaction.

- Completed: Phase 0 diagnostic reporting exists for processed datasets, encoded manifests, token distributions,
  reference comparisons, MLflow summaries, and tonal probes.
- Completed: Phase 1A introduced `metadata.tokenization_context` with separate pitch-set basis, declared key hint,
  spelling key fifths, and spelling-context source. Encoded manifests and diagnostics expose spelling context.
- Completed: Phase 1B made notation decoding use pitch-set metadata for MIDI recovery and `spelling_key_fifths` for
  displayed key signatures and visible accidentals.
- Completed: Phase 1C made `metadata.tokenization_context` required, removed internal compatibility fallbacks, and
  invalidates stale encoded artifacts by bumping the tokenizer schema version after the tokenization-context metadata
  change.
- Completed: Phase 1D added focused notation and MusicXML spelling fixtures for harmonic minor, melodic minor,
  borrowed tones, and modal spelling.
- Completed: Phase 2A added a lossless factorized token representation, derived factorized target tensors during
  dataset example construction, and logs flat-logit attribute accuracies for duration, degree, accidental, register,
  and hand.
- Next: Phase 2B should add factorized model heads and masked per-attribute losses while keeping the flat objective
  runnable as a baseline.

Early non-unit validation for Phase 2A:

```bash
DATA_DIR=data/exercises PROCESS_DISABLE_MLFLOW=1 PROCESS_SKIP_FIGURE_ANALYSIS=1 \
  PROCESS_WHOLE_FILE_SEGMENTS=1 PROCESS_TOKENIZATION_WORKERS=1 PROCESS_TOKENIZATION_BATCH_SIZE=1 \
  PROCESS_OVERWRITE=1 make process

uv run python scripts/pretrain.py --data-dir data/exercises --whole-file-segments \
  --epochs 1 --batch-size 2 --device cpu --num-workers 0 \
  --checkpoint-dir /tmp/musak-factorized-pretrain --disable-mlflow --overwrite --no-progress
```

The one-epoch run should emit non-null duration, degree, accidental, octave-offset, and hand accuracies from the flat
token logits. With MLflow enabled, these appear under `model/<train|validation>/rate/*_accuracy`.

## Phase 0: Baseline Audit

Goal: make the current failure modes visible before changing schemas.

Implementation tasks:

- Add a small diagnostic command or notebook helper that reports scale-match outcomes by source file, declared key,
  selected root/type, low-confidence status, accidental fraction, and in-scale fraction.
- Add focused fixtures for major, natural minor, harmonic minor, melodic minor mixture, borrowed tones, and missing or
  wrong key signatures.
- Capture current training/generation baseline metrics for the same dataset and config:
  flat validation loss, token-kind accuracy, invalid-target rate, hard-constrained generation completion, figure
  distance, rhythm metrics, accidental fraction, and in-scale fraction.

Acceptance criteria:

- We have a reproducible baseline report.
- Minor-key and modal examples distinguish intentional pitch-set collapse from spelling or chromatic-context errors.
- No model or tokenizer behavior changes yet.

Likely code areas:

- `musak_model/data/scale_matcher/`
- `musak_model/processing/manifest.py`
- `tests/musak_model/data/scale_matcher/`
- `tests/musak_model/data/test_converter.py`

## Phase 1: Scale Basis And Spelling Semantics

Goal: keep pitch-set token semantics while making notation spelling and chromatic context explicit.

Implementation tasks:

- Introduce a tokenization-context decision object. It should distinguish:
  - pitch-set scale basis used for token coordinates;
  - optional tonal or modal spelling hint;
  - declared key-signature hint;
  - pitch-set support diagnostics.
- Keep natural-minor and modal pitch-set collapse intentional unless we explicitly choose to add a mode-conditioned
  token basis later.
- Improve harmonic-minor, melodic-minor, borrowed-tone, and modal spelling so accidentals are driven by tokenization
  context and declared orthography, not only by the sign of key-signature fifths.
- Keep pitch-set matching as diagnostics and filtering support, and make reports explicit about pitch-set basis versus
  tonal or modal spelling.
- Update tokenizer snapshots and encoded artifact compatibility checks so old artifacts cannot silently train the new
  semantics.
- Update `docs/model.md` to describe the new semantics and migration requirement.

Acceptance criteria:

- Natural-minor and modal collapse is explicit in diagnostics and not reported as an error by itself.
- Harmonic and melodic minor examples preserve expected scale-family coordinates and accidentals.
- Borrowed tones and modal examples preserve plausible spelling when declared notation context gives a clear hint.
- Wrong or missing key signatures can still be overridden by observed notes for pitch-set basis selection.
- MusicXML/notation round trips preserve plausible spelling for major, minor, and modal examples.

Likely code areas:

- `musak_model/data/scale_matcher/`
- `musak_model/data/converter.py`
- `musak_model/data/segmenter/bar.py`
- `musak_model/tokens/schema.py`
- `musak_model/tokens/pitch.py`
- `musak_model/decoder/notation.py`
- `musak_model/processing/snapshot.py`
- `tests/musak_model/data/`
- `tests/musak_model/decoder/`
- `tests/musak_model/validation/`

## Phase 2: Factorized Token Representation And Diagnostics

Goal: make the current flat vocabulary decomposable into musical attributes before changing the model objective.

Implementation tasks:

- Add a lossless `TokenAttributes` representation derived from existing `Token` objects:
  - token kind;
  - note degree;
  - accidental;
  - octave offset;
  - duration;
  - hand where applicable.
- Reconstruct tokens and flat token ids from strict factorized targets.
- Reconstruct tokens and flat token ids from model-style factorized predictions by using the predicted kind to select
  active attributes.
- Derive factorized target tensors during dataset example construction without changing encoded artifact shape.
- Add flat-logit diagnostics that report attribute accuracies before the factorized objective exists.

Acceptance criteria:

- Every current flat vocabulary id factorizes and reconstructs exactly.
- Malformed strict targets fail fast.
- Dataset examples and batches carry factorized targets with an explicit absent-attribute id for inactive heads and
  padding.
- Training metrics separate duration, degree, accidental, octave-offset, and hand failures even when the model still
  trains with the flat softmax objective.

Likely code areas:

- `musak_model/tokens/factorized.py`
- `musak_model/training/dataset/examples.py`
- `musak_model/training/dataset/collate.py`
- `musak_model/training/dataset/schema.py`
- `musak_model/training/metrics/`
- `tests/musak_model/tokens/`
- `tests/musak_model/training/dataset/`
- `tests/musak_model/training/metrics/`

## Phase 3: Factorized Event Objective

Goal: reduce data demand by predicting structured event attributes instead of only one flat vocabulary ID.

Implementation tasks:

- Add a `TokenAttributeTarget` representation derived from existing `Token` objects:
  - token kind;
  - note degree;
  - accidental;
  - octave offset;
  - duration;
  - hand where applicable.
- Add a factorized model head:
  - token-kind head for all positions;
  - duration head for notes, rests, and holds;
  - degree, accidental, and octave heads for notes;
  - optional join/bar/end/hand heads as categorical attributes only if the simple kind head is not enough.
- Keep the flat vocabulary head available behind config until the factorized path is validated.
- Add masked cross-entropy losses per active attribute. Do not train note attribute heads on rest, hold, bar, hand, join,
  or end targets.
- Add explicit loss weights in config and MLflow logging for every objective term. The first combined objective should
  be:

  ```text
  loss = event_kind_ce
       + duration_weight * duration_ce
       + degree_weight * degree_ce
       + accidental_weight * accidental_ce
       + octave_weight * octave_ce
       + hand_weight * hand_ce
  ```

  Start with conservative equal weights for active heads, then tune only after per-head metrics show which objective
  dominates.
- Add attribute metrics:
  - token-kind accuracy;
  - duration accuracy;
  - degree accuracy;
  - accidental accuracy;
  - octave accuracy;
  - exact reconstructed-token accuracy.
- Add a reconstruction helper that converts factorized predictions back to token IDs for constrained sampling.

Acceptance criteria:

- Factorized validation loss and reconstructed-token accuracy are comparable to or better than the flat baseline.
- Attribute metrics reveal whether failures are mostly rhythmic, pitch-class, accidental, or register errors.
- Per-head losses are logged separately, so a low total loss cannot hide collapsed accidentals, rhythm, or register.
- Hard-constrained sampling works from factorized logits.
- Flat-token training remains runnable as a baseline.

Likely code areas:

- `musak_model/tokens/`
- `musak_model/model/hierarchical.py`
- `musak_model/model/config.py`
- `musak_model/training/stages/pretraining.py`
- `musak_model/training/metrics/`
- `musak_model/evaluation/generation/`
- `tests/musak_model/tokens/`
- `tests/musak_model/training/`
- `tests/musak_model/evaluation/`

## Phase 4: Bar-Relative Musical Coordinates

Goal: stop requiring the model to infer musical time only by accumulating previous duration tokens.

Implementation tasks:

- Add per-token bar-relative position metadata for training. Start by deriving it from token streams during dataset
  example construction; persist it in encoded artifacts only after the shape is stable.
- Represent the coordinate as integer ticks over the duration vocabulary denominator, with a no-position bucket for
  tokens that do not advance musical time.
- Add embeddings for bar-relative position and active hand state, or feed these as a compact structured input alongside
  token embeddings.
- Keep hard generation constraints as the authority for exact measure validity.

Acceptance criteria:

- The model input has explicit bar position information for note, rest, and hold decisions.
- Validation metrics separate timing/duration failures from pitch/register failures.
- Generation does not regress on hard-constrained bar completion.

Likely code areas:

- `musak_model/training/dataset/examples.py`
- `musak_model/training/dataset/collate.py`
- `musak_model/training/dataset/schema.py`
- `musak_model/model/hierarchical.py`
- `musak_model/model/config.py`
- `tests/musak_model/training/dataset/`
- `tests/musak_model/model/`

## Phase 5: Musical Auxiliary Objectives

Goal: make the model predict exercise-level musical properties that are not visible in token perplexity.

Implementation tasks:

- Define an `ExerciseAttributeTarget` or equivalent training-side structure derived from encoded samples and existing
  diagnostics. Start with targets that can be extracted reliably:
  - token count bucket;
  - note density bucket;
  - onset density bucket;
  - maximum simultaneous notes bucket;
  - hand activity or texture bucket;
  - register center bucket;
  - static hand span bucket;
  - maximum melodic gap bucket;
  - in-scale/chromatic fraction bucket;
  - rhythm profile bucket, using the same beat and duration grids as figure-profile artifacts;
  - difficulty label when the finetuning dataset provides one.
- Add sequence-pooled and optional bar-pooled heads for these targets. These heads should train on teacher-forced
  references and should not affect constrained decoding until their metrics are understood.
- Keep harmonic objectives conservative at first:
  - measure consonance and dissonance rates from vertical pitch sets;
  - measure scale-degree stability at bar endings and phrase endings;
  - add chord-context labels only after a reliable chord or harmony extractor exists.
- Log auxiliary loss and accuracy per target. Do not collapse all musical targets into one uninspectable
  "musicality" scalar.
- Add generated-sample reports for the same attributes. The key question is not only whether the model can classify a
  reference sample, but whether sampled exercises land near the exercise reference distribution.
- Gate training experiments on these reports. If generated samples still match PDMX density and texture more than the
  exercise set, the loss is not aligned with the product goal.

Acceptance criteria:

- Auxiliary targets can be computed deterministically from encoded artifacts.
- Training logs include per-target losses and accuracies.
- Generated-sample diagnostics compare model output against both PDMX and exercises for density, rhythm, register,
  hand activity, chromaticity, and difficulty.
- No auxiliary target is used as a generation reward until it has a reference distribution and a failure-mode test.

Likely code areas:

- `musak_model/processing/diagnostic_report.py`
- `musak_model/training/dataset/examples.py`
- `musak_model/training/dataset/schema.py`
- `musak_model/training/metrics/`
- `musak_model/training/`
- `musak_model/model/`
- `tests/musak_model/training/`
- `tests/musak_model/processing/`

## Phase 6: Reference Priors And Reranking

Goal: steer samples toward exercise-like local structure without turning exact n-grams into model tokens.

Implementation tasks:

- Define a low-cardinality `FigureFamily` from existing figure properties:
  contour shape, duration shape, monophonic/chordal flags, in-scale flag, opening interval class, closing interval
  class, and figure length.
- Add transition counts over ordered figure occurrences per sample and hand.
- Add conditional reference counts keyed by family, hand, scale type, metrical position, anchor bin, base duration, and
  optional chord context.
- Export new artifacts alongside existing figure profile artifacts without changing current `figure/all/counts.parquet`
  semantics.
- Add deterministic backoff queries for generation and evaluation.
- Add a generated-sample scorer that combines interpretable terms:
  - hard validity failures;
  - rhythm distribution distance;
  - density and hand-texture distance;
  - register and span distance;
  - figure-family transition surprise;
  - chromaticity and scale-degree stability;
  - harmonic-consonance or chord-context surprise when available.
- Use reference priors first for reranking and shallow fusion, not for end-to-end training. This keeps non-differentiable
  music rules inspectable and avoids reward hacking before the supervised model is competent.
- Keep all priors conditional on difficulty, bar count, meter, and hand when enough data exists. Back off deterministically
  when counts are sparse.

Acceptance criteria:

- Existing figure-profile metrics remain unchanged.
- New artifacts can answer:
  `P(family_t | family_{t-1}, hand, meter_position, chord_context)` and
  `P(figure | family, scale_type, hand, n, anchor_bin, meter_position)`.
- Missing or partial artifacts fail fast.
- Backoff order is deterministic and tested.
- Reranking can improve generated-sample diagnostics without changing model weights.
- Any prior that rejects a sample can explain which term caused the rejection.

Likely code areas:

- `musak_model/n_grams/figure/`
- `musak_model/n_grams/profile/`
- `musak_model/training/stages/figure_profiles/`
- `musak_model/evaluation/generation/figure_metrics.py`
- `tests/musak_model/n_grams/`
- `tests/musak_model/training/stages/`

## Phase 7: Planner And Plan-Conditioned Generation

Goal: move exercise structure out of raw next-token prediction.

Implementation tasks:

- Train or fit a low-entropy planner over bar count, phrase role, density, hand activity, register bin, harmonic
  context, figure family, rhythm profile, and base duration.
- Decode plans through the empirical figure sampler and deterministic renderer first.
- Add a neural token decoder only if the renderer baseline fails on surface quality.
- If a token decoder is added, condition it on per-bar or per-cell plan states, not only on one prefix vector.
- Add shallow-fusion support for reference priors:
  `logits = model_logits + alpha * log_reference_prior + beta * soft_penalties`.

Acceptance criteria:

- Planner or renderer generations beat the current neural baseline on hard validity, rhythm, register, harmony, and
  figure-family metrics.
- User controls reliably affect the generated exercise.
- The token decoder is optional and does not weaken plan adherence.

Likely code areas:

- `musak_model/synthetic/`
- `musak_model/harmony/`
- `musak_model/evaluation/generation/`
- `musak_model/model/`
- `musak_model/training/`

## Phase 8: Small-Data Training Hygiene

Goal: improve training efficiency after representation and objective terms are measurable.

Implementation tasks:

- Implement conditioning dropout if `cfg_dropout_probability` remains part of the config.
- Add early stopping or best-checkpoint selection based on validation loss plus generated-sample diagnostics.
- Audit unused augmentation helpers and either wire useful ones into training intentionally or remove them from the
  training story.
- Compare model shapes only after Phases 3 and 4 are available:
  - flat baseline;
  - factorized Transformer only;
  - factorized model with musical auxiliary heads;
  - factorized model with the existing CNN/GRU hierarchy enabled, if the simpler model is still underfitting.
- Treat PDMX as broad pretraining data and exercises as the target distribution. Use exercise-shaped PDMX filtering or
  curriculum weighting before full-corpus training.
- Keep model capacity conservative. Prefer better targets, coordinates, and priors over a larger Transformer.

Acceptance criteria:

- Training logs include enough metrics to compare flat, factorized, auxiliary, and prior-guided runs fairly.
- Conditioning dropout behavior is tested and documented.
- Any augmentation used in training has tests proving that timing, register limits, and token validity are preserved.
- The selected default config is justified by validation and generation metrics, not by training loss alone.
- A model is not promoted unless generated samples improve over the flat baseline on exercise-domain diagnostics.

Likely code areas:

- `musak_model/training/`
- `musak_model/model/`
- `musak_model/data/augmentation.py`
- `musak_model/configs/training/`
- `musak_model/configs/model/`
- `tests/musak_model/training/`
- `tests/musak_model/data/test_augmentation.py`

## Immediate Pull Request Sequence

1. Finish Phase 0 diagnostic fixtures and baseline reporting, including generated-sample reports where possible.
2. Implement Phase 1 tokenization-context and spelling semantics with tests and docs.
3. Finish Phase 2 factorized targets, reconstruction helpers, and metrics without changing model training.
4. Add Phase 3 factorized model heads and per-head losses behind config.
5. Add Phase 5 auxiliary musical target extraction and dataset reports without changing generation.
6. Add Phase 4 bar-relative coordinates if timing metrics show accumulation errors, or earlier if the target schemas need
   bar pooling.
7. Add Phase 6 reference-prior scoring and reranking for generated samples.
8. Only then run serious PDMX/exercise training comparisons.

## Non-Goals For This Cycle

- Do not directly port Moonbeam's MIDI tokenizer. Musak is score-based, piano-focused, scale-relative, and much smaller.
- Do not add exact n-gram IDs to the neural vocabulary.
- Do not rely on validation cross-entropy alone as a musicality signal.
- Do not train a full model just to see whether the current objective works. The objective and diagnostics come first.
- Do not introduce adversarial training, RL, or a learned reward model until supervised factorized and auxiliary
  objectives have a strong baseline and inspectable failure modes.
- Do not collapse musical quality into one scalar loss before individual rhythm, register, texture, chromatic, and
  harmonic terms are logged and ablated.
- Do not mix synthetic data into neural training until the stochastic generator is calibrated against reference metrics.
