# Musical Coherence Improvement Plan

This plan extends the tokenization/training roadmap with a concrete path toward more coherent melody, harmony, and
two-hand texture. Follow [guidelines.md](guidelines.md) before implementing each phase, and update
[model.md](model.md) whenever a phase changes training inputs, generation behavior, metrics, or artifacts.

## Problem Statement

The current model can learn much of the local token grammar, especially under hard generation constraints, but it does
not reliably produce musical exercises. Recent pretrain and finetune runs show legal samples with weak melodic purpose,
bar-long stasis, coarse harmonic behavior, and limited interaction between hands. The likely cause is not only model
size. The autoregressive token objective still asks the decoder to infer phrase role, cadence, texture, and figure
reuse from a small amount of relevant data.

The goal is to expose and condition on lower-entropy musical structure before relying on larger models or reward
training.

## Design Principles

1. Keep hard legality constraints separate from musicality.
2. Make bad musical behavior visible in metrics before changing training.
3. Plan harmony globally, not as an aimless pair transition.
4. Treat chord pair transitions as one score term inside a finite-horizon planner, not as the planner itself.
5. Tell the model where it is in the phrase and how close it is to the ending.
6. Validate figure and texture abstractions as diagnostics and reranking features before making them core model inputs.
7. Prefer reranking and supervised auxiliary targets before reinforcement learning or scalar musicality rewards.

## Phase 0: Musical Coherence Diagnostics

Goal: make the failure modes audible in metrics.

Status: in progress. Core generation diagnostics are implemented under
`generation/<soft|hard>/coherence/*`; reranking use is still deferred.

Implementation tasks:

- Add generation diagnostics for whole-bar and long-duration stasis by hand.
  Explicit whole-note-or-longer metrics are tracked separately from whole-bar metrics, because short bars and non-4/4
  meters make those concepts diverge.
- Add melodic-contour diagnostics: stepwise motion, large leaps, leap recovery, repeated notes, and direction changes.
- Add hand-dialogue diagnostics: synchronized onsets, delayed answer onsets, contrary motion, and static long-bass
  support under upper-hand motion.
- Add phrase-closure diagnostics: final bass support, final melody degree, final long-note closure, and final-slot
  activity.
- Keep these metrics diagnostics only. Do not add them to the sample penalty until they distinguish good and bad
  samples on manual inspection.

Acceptance criteria:

- Recent bad hard-constrained samples show elevated long-duration/stasis or weak dialogue metrics.
- Metrics are logged under a separate namespace suitable for future reranking.
- Unit tests cover at least one static-bass sample and one simple call-and-response sample.

## Phase 1: Form-Aware Harmonic Planner

Goal: replace blind local chord sampling with a finite-horizon harmonic plan.

Implementation tasks:

- Introduce harmonic slot roles: opening, continuation, cadence preparation, cadence.
- Add distance-to-end and final-slot requirements to the harmonic planner.
- Score whole chord plans with:
  - local chord prior;
  - pair-transition smoothness;
  - phrase-role compatibility;
  - cadence target score;
  - repetition and variation preference;
  - optional empirical transition score when fitted artifacts are available.
- Generate the whole plan with dynamic programming or beam search over harmonic slots.
- Keep the existing functional pair model as a backoff score, not as the top-level planner.

Acceptance criteria:

- Two-bar plans strongly prefer sensible cadential shapes over arbitrary `iii -> ii` style motion.
- Four-, eight-, and sixteen-bar plans expose phrase role and distance-to-end in artifacts.
- The dataset-quality notebook can display role-labeled harmonic plans.

## Phase 2: Sub-Bar Harmonic Rhythm

Goal: make harmony more expressive without overfitting noisy chord decoding.

Implementation tasks:

- Test `chord_decoding.resolution: 2` before considering denser settings.
- Inspect decoded half-bar plans in the dataset-quality notebook.
- Compare plan agreement, strong-beat chord-tone coverage, bass support, and final-slot closure.
- Keep sub-bar windows aligned through existing decoder coordinates.

Acceptance criteria:

- Half-bar plans improve harmonic shape without producing visibly noisy chord flips.
- Samples do not become literal chord-tone exercises.

## Phase 3: Candidate Reranking

Goal: improve generation quality without changing training loss.

Implementation tasks:

- Generate multiple hard-valid candidates per request.
- Rank candidates with a transparent score built from Phase 0 and harmony metrics:
  - cadence closure;
  - bass harmonic support;
  - strong-beat harmonic clarity;
  - melodic contour;
  - hand dialogue;
  - figure reference similarity;
  - penalties for whole-bar stasis, stuck repetition, and harsh coincident clashes.
- Log the full term breakdown for every candidate.

Acceptance criteria:

- Reranked top samples are audibly better than random hard samples often enough to justify using the reranker by
  default.
- No reranking term encourages bland all-chord-tone output.

## Phase 4: Phrase And Form Conditioning

Goal: make the decoder aware of phrase position and endings.

Implementation tasks:

- Add low-cardinality conditioning IDs for phrase position, bar role, cadence strength, and distance-to-end.
- Align these controls per decoder step similarly to harmony IDs.
- Add optional bar-count and phrase-role controls to generation.
- Add auxiliary targets for phrase/closure behavior when derivable from data.

Acceptance criteria:

- The model receives explicit end-of-phrase context instead of inferring endings only from prefix length.
- Generation metrics improve closure without reducing local rhythmic variety.

## Phase 5: Inspectable Figure And Texture Plan

Goal: validate a non-neural abstraction for melody and texture before committing to architectural changes.

Implementation tasks:

- Derive figure families from existing n-gram artifacts:
  - rhythm shape;
  - contour shape;
  - interval pattern;
  - start/end degree relation;
  - chord-relation profile;
  - hand;
  - slot position;
  - duration class.
- Derive texture roles: melody, bass support, accompaniment, echo, sparse rest.
- Use these labels for diagnostics, reference distributions, reranking terms, and optional auxiliary targets.

Acceptance criteria:

- Figure/texture families are stable enough to distinguish sight-reading exercises from rambling samples.
- The abstraction remains inspectable in notebooks and artifacts.

## Phase 6: Anchored Figure Embeddings

Goal: give the neural model a reusable representation of musical gestures.

Implementation tasks:

- Define anchored figures with hand, start slot, duration span, anchor degree, register, rhythm family, contour family,
  and chord-relation family.
- Initialize figure similarity with explicit inductive bias:
  - transposed contours are close;
  - shared rhythm shapes are close;
  - shared chord-relation profiles are close;
  - exact pitch identity is secondary.
- Train or fine-tune embeddings with supervised auxiliary, contrastive, or retrieval-style objectives.

Acceptance criteria:

- Similar figures cluster musically, not merely by flat token identity.
- The embedding improves figure retrieval or reranking before it is used as a mandatory decoder input.

## Phase 7: Figure-Conditioned Decoder

Goal: decode notes as realizations of planned figures rather than isolated next-token choices.

Implementation tasks:

- Add active-figure conditioning to the decoder:

  ```text
  p(next token | prefix, coordinates, harmony, phrase, texture, active figure)
  ```

- Generate in two stages:
  1. plan harmony, phrase, texture, and anchored figures;
  2. decode tokens under that plan.
- Keep hard legality constraints active.

Acceptance criteria:

- Melody shows more stable local gestures and controlled variation.
- Hands exhibit planned roles instead of independent rambling.

## Phase 8: CNN/GRU Bar-Memory Ablation

Goal: test whether existing local and bar-level modules improve coherence once planning signals exist.

Implementation tasks:

- Compare current transformer, CNN-only, GRU-only, and CNN+GRU with the same data and conditioning.
- Track validation, generation, Phase 0 diagnostics, and manual listening notes.
- Keep the simplest architecture that improves generation.

Acceptance criteria:

- CNN/GRU is kept only if it improves musical behavior, not merely teacher-forced loss.

## Phase 9: Preference Or Reward Training

Goal: use rewards only after metrics and reranking are proven useful.

Implementation tasks:

- Collect preference pairs from generated samples.
- Prefer DPO-style or supervised preference prediction before RL.
- Avoid direct optimization of a vague scalar musicality reward until it is robust against metric gaming.

Acceptance criteria:

- Reward or preference training improves blind manual comparisons without degrading legality or variety.

## Immediate Implementation Order

1. Phase 0 diagnostics.
2. Phase 1 form-aware harmonic planner.
3. Phase 2 half-bar harmonic rhythm.
4. Phase 3 reranking.
5. Phase 4 phrase/form conditioning.
6. Reassess before committing to Phase 5 figure planning and Phase 6 anchored embeddings.
