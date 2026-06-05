# Exercise Metrics

This document defines the planned metric families for evaluating sight-reading exercises. The same conceptual metrics
should apply to processed datasets and generated samples whenever they can be computed from a tokenized `Segment`.

Metrics should stay descriptive at first. Structural and playability failures are quality signals; style metrics such as
syncopation, density, register, or hand interleaving are artistic or difficulty choices unless they exceed explicit
constraints. Avoid a single aggregate quality score until the separate distributions are understood.

## Metric Families

### Structural Validity

- Decode success: whether a token stream can be interpreted as musical events.
- Bar alignment: whether hand cursors align with bar boundaries.
- End alignment: whether final hand cursors align with expected segment duration.
- Invalid transitions: illegal `JoinWithPreviousToken`, `HoldToken`, `BarToken`, or `EndToken` use.
- Silent and sparse structure: empty score, one-hand-only, silent bars, silent edge bars.
- Duration grid compliance: whether event starts and durations stay on the configured rhythmic grid.

### Tonality and Scale

- Accidental note fraction and in-scale note fraction.
- Duration-weighted accidental or in-scale fractions.
- Explained and unexplained chromaticism, especially for harmonic and melodic minor mixtures.
- Pitch-class entropy and observed pitch-class count.
- Scale-match confidence, ambiguity, and no-pitch diagnostics.

### Figures

- Common figure mass: generated figure occurrences found among dataset-common figures.
- Rare figure mass: occurrences whose dataset count is below a threshold.
- Novel figure mass: occurrences absent from the reference dataset.
- Figure distribution distance, grouped by scale type, hand, and n-gram length.
- Figure properties: monophonic, chords-only, in-scale.
- Contour distribution and normalized duration-shape distribution.

Current implemented figure comparisons cover aggregate property-rate errors and direct figure identity distribution
distance against canonical `figure/all/counts.parquet` artifacts. Training generation evaluation logs these when figure
artifacts are available, and the model output explorer shows generated-output figure metrics for notebook inspection.

### Rhythm and Meter

- Note density and onset density per beat or bar.
- Shortest note duration and dotted-note presence.
- Duration entropy and rhythmic-value distribution.
- Grid alignment at beat, half-beat, and subdivision levels.
- Strong-beat onset fraction.
- Syncopation index.
- Rhythmic n-gram distribution.
- Whole-bar note and whole-note-or-longer event rates.
- Long sustain without opposite-hand answer rate.
- Per-hand activity labels over a planning grid: rest, onset, sustain.
- Coactivity-mode distribution: silent, right-only, left-only, synchronized, answering, interleaved, both-sustain.
- Learned-salience onset alignment and syncopation salience.
- Onset-template and duration-profile distribution distance against reference slices.

### Playability

- Maximum same-hand notes per onset.
- Maximum same-hand chord span in semitones.
- Maximum same-hand melodic gap in semitones.
- Static hand span in scale degrees.
- Pitch range and register extremes by hand.
- Hand crossing rate.
- Maximum simultaneous note count across both hands.

### Hand Coordination

- Right, left, both-hands, and single-hand active fractions.
- Hand activity balance.
- Synchronized onset fraction.
- Independent onset fraction.
- Hand interleaving rate.
- Left/right density ratio.
- Role separation by register, density, and chord rate.

### Pitch Style

- Pitch mean, variance, and range by hand.
- Melodic interval distribution.
- Melodic direction distribution.
- Direction change rate.
- Stepwise motion fraction.
- Repetition rate.
- Register drift over bars.
- Climax position.

### Structure and Repetition

- Bar-level density curve.
- Motif recurrence.
- Self-similarity between bars or phrase windows.
- Opening and closing register contrast.
- Cadential simplicity proxies.
- Exact-repeat, varied-repeat, answer, contrast, and cadence-fill relation rates.
- Bar-pair and phrase-pair rhythm similarity matrices.
- Recurrence without stasis rate.
- Plan-following metrics when rhythm or figure plans are supplied: onset precision, onset recall, hand-activity
  accuracy, duration-profile agreement, and plan escape rate.

See [rhythm-figure-plan.md](rhythm-figure-plan.md) for the planned rhythm/texture/figure metric definitions and rollout.

## Rollout

### V0: Segment Diagnostics

V0 metrics are computed from a tokenized `Segment` plus a `DurationVocabulary`, without a reference dataset. They are
the common ground for dataset processing and generation evaluation.

- Existing structural and activity diagnostics: silence fractions, both-hands activity, silent bars, hand activity
  balance, empty score, one-hand-only, and token-kind fractions.
- Tonality: accidental note fraction and in-scale note fraction.
- Rhythm and density: note density per beat, onset density per beat, per-hand onset density per beat, shortest note
  duration in beats, and dotted-note presence.
- Playability: maximum notes per onset, maximum notes per hand, maximum onset span, maximum melodic gap, and static hand
  span.
- Hand coordination: synchronized onset fraction and independent onset fraction.

### V1: Dataset-Relative Distributions

V1 compares generated output with reference dataset distributions. Comparisons should be sliced by compatible metadata
where possible: scale type, time signature, hand, and figure n-gram length.

The implemented V1 slice currently covers figure profile property comparisons and figure identity total variation
distance. These are descriptive metrics only; they do not constrain generation.

- Common, rare, and novel figure mass.
- Figure distribution distance.
- Figure property distributions.
- Contour and duration-shape distributions.
- Rhythmic n-gram distributions.
- Duration entropy and grid-alignment distributions.
- Strong-beat onset fraction.

### V2: Musical Style and Difficulty Shape

V2 metrics describe style, difficulty, and texture with more musical interpretation.

- Syncopation index.
- Hand interleaving rate.
- Role separation between hands.
- Pitch range and pitch variance by hand.
- Melodic interval distributions.
- Stepwise motion fraction.
- Direction change rate.
- Register extreme fraction.
- Hand crossing rate.
- Vertical density and simultaneous note count.

### V3: Higher-Level Structure

V3 should be implemented only after V0-V2 distributions are understood and manually checked against examples.

- Motif recurrence.
- Bar-level density curves.
- Self-similarity across bars or phrase windows.
- Cross-hand answer rate and best lagged phrase similarity.
- Recurrence without stasis.
- Rhythm and figure relation-label distributions.
- Opening and closing contrast.
- Climax position.
- Cadential simplicity proxies.
- Composite quality or difficulty scores.

## Naming

Diagnostic field names should be domain-agnostic because the same values can be logged for datasets and generation.
MLflow names can still use the current `dataset/...` and `generation/...` domains while those run types remain separate.
If generation evaluation later becomes a general analysis command, the metric domain prefix can be revisited without
changing the shared diagnostic model.
