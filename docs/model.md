# Musak Model Pipeline and Controllable Generation

## Current Encoded Dataset

Processed datasets are stored under:

```text
processed/<dataset-name>/
  parsed.csv
  parsed/<source-hash-prefix>/<source-hash>.json
  encoded/<tokenizer-hash>/
    tokenizer.json
    encoded.csv
    data-00000.jsonl
```

Training receives explicit dataset directories. With `--processed-dir processed/PDMX`, the training code looks for reusable encoded artifacts under `processed/PDMX/encoded/<tokenizer-hash>`. `--data-dir data/PDMX` is optional when processed artifacts are usable, and is required only for raw MusicXML fallback. When both directories are supplied, dataset names must match. Encoded artifacts are used only when `tokenizer.json` matches the active tokenizer snapshot. Otherwise, training falls back to parsed JSON, and then raw MusicXML only if `--data-dir` was supplied.

Each encoded JSONL row is an `EncodedExercise`:

- `token_ids`: unified two-hand token sequence.
- `bar_positions`: one bar index per token for hierarchical model context.
- `metadata`: scale root, scale type, time signature, source file, segment window, optional difficulty label, and extracted difficulty features.

The current encoded PDMX artifact is a unified two-hand stream. It is not split into separate right-hand and left-hand samples.

Data processing uses `DataProcessingConfig.remove_segments_with_silent_bars=True` by default. Segmentation still
records silent-bar diagnostics, but processing marks any segment with a fully silent bar as ineligible with
`silent_bar` before writing encoded samples. Edge-silent windows keep the more specific `silent_edge_bar` reason too.

## Current Token Semantics

Tokens represent music relative to segment metadata:

- `NoteToken`: scale degree, accidental, octave offset, and duration.
- `RestToken`: active-hand rest duration.
- `HoldToken`: active-hand continuation duration for the previous same-hand note or chord.
- `HandToken`: switches the active hand between right and left.
- `JoinWithPreviousToken`: joins a note to the previous onset for chord representation.
- `BarToken`: bar boundary.
- `EndToken`: sequence end.
- `StartToken`: training/generation beginning-of-sequence token.

Scale root is external metadata. It names the root of the pitch set used for tokenization, not necessarily the
tonal tonic. For example, natural minor and modes are represented by the corresponding `major` pitch set root.
Because pitches are stored as scale degrees, the same token stream can be decoded in different roots by changing
`metadata.scale_root`.

Scale type is also metadata, but it affects pitch-class decoding. The supported scale families are `major`,
`harmonic_minor`, and `melodic_minor`. Modes are intentionally collapsed into the `major` pitch-set family.

During processing, raw parsed windows are matched to `scale_root` and `scale_type` before tokenization. The matcher
uses duration-weighted pitch-class distributions over the segment. MusicXML key signatures are retained as declared
hints and diagnostics; they are not trusted as the tokenization source of truth.

Declared key signatures are stored as `declared_key_fifths`, a circle-of-fifths coordinate. Scale roots are stored as
pitch classes in semitones from C. Conversion between these coordinate systems is owned by `musak_shared.elements`:
`pitch_class_from_key_fifths(key_fifths) = (key_fifths * 7) % 12`. The reverse conversion chooses the enharmonic
spelling with the simpler key signature.

`JoinWithPreviousToken` is not a tie token. It represents simultaneous note onsets, usually chord notes. True prolongation is represented by `HoldToken`.

## Scale Matching Procedure

The processing pipeline cannot rely on MusicXML key signatures as tokenization truth. Some scores omit key signatures
and spell the pitch set with accidentals, some have incorrect declarations, and modal music often uses a major or
minor key signature plus accidentals even though tokenization needs the actual pitch set. The goal is therefore to
infer the segment's pitch-set root and scale family from the notes themselves, while keeping declared key signatures
only as diagnostics and tie-break hints.

Scale matching runs per segment window, before tokenization. This matters because tokenization converts MIDI pitches
to scale degrees and accidentals; if the wrong scale is chosen first, the token sequence is already distorted. The
matcher assumes one stable pitch set per segment. If a segment modulates without an explicit key-signature change, the
matcher will choose the best single explanation and expose the uncertainty through metrics.

The candidate space is intentionally small:

- `scale_root`: every pitch class from 0 to 11;
- `scale_type`: `major`, `harmonic_minor`, and `melodic_minor`.

Natural minor and modes are represented by the `major` pitch-set family because they contain the same pitch classes as
some major scale. For example, A natural minor maps to `scale_root=0, scale_type=major` because its pitch set is C
major. This is a pitch-set choice, not a claim that the tonal tonic is C.

For each segment, the matcher builds a duration-weighted pitch-class histogram from both hands:

- a note contributes its duration to `midi_pitch % 12`;
- each chord pitch contributes the chord duration to its own pitch class;
- rests do not contribute.

Each candidate scale defines seven pitch classes. Its score is:

```text
in_scale_weight_fraction = duration weight inside candidate pitch classes / total pitched duration weight
```

The best candidate is the one with the highest `in_scale_weight_fraction`. If multiple candidates tie for the best
score and the declared key-signature pitch set is one of those tied candidates, the declared candidate is selected.
Otherwise, selection is deterministic: prefer the configured scale-family order, then the simpler key-signature root,
then the lower numeric root. This tie policy prevents random dataset churn while making declared metadata useful only
when the notes do not disambiguate the scale.

The matcher records these diagnostics for every encoded-manifest row:

- `in_scale_weight_fraction` and `out_of_scale_weight_fraction`;
- `best_margin`, the score gap between the selected candidate and the next distinct score;
- `observed_pitch_class_count`;
- `tied_best_candidate_count`;
- `declared_match_used`;
- `low_confidence`, `ambiguous`, and `no_pitches`.

Processing can mark segments ineligible based on configurable thresholds. The default policy excludes weak matches:
`minimum_in_scale_weight_fraction = 0.90` and `minimum_best_margin = 0.03`. A segment is marked
`scale_match_low_confidence` when too much duration falls outside the selected scale, when the best margin is too
small, or when there are no pitches. A segment is marked `scale_match_ambiguous` when multiple best candidates tie and
the declared key signature does not resolve the tie. A segment with no pitched events is also marked
`scale_match_no_pitches`.

This procedure is intentionally not a full tonal analysis. It does not infer cadences, tonic function, modulation,
borrowed harmony, or enharmonic spelling intent. It only chooses a stable pitch-set basis for scale-relative
tokenization. Non-standard and atonal material may be forced into the least-bad candidate or filtered by confidence
thresholds. The quality of this decision should be monitored with the scale-match MLflow metrics and manually through
the dataset statistics notebook.

## Current Training Logic

`PretrainingTrainer` trains an autoregressive next-token model with teacher forcing:

- dataset rows are converted to input and target token IDs;
- `StartToken` is prepended to the model input;
- targets remain the original musical tokens;
- batches pad token IDs and bar positions;
- loss ignores padded target positions.

The model is hierarchical:

- token embeddings feed a local convolution encoder;
- bar-local representations are encoded with GRUs;
- a causal Transformer decoder predicts next-token logits;
- bar context is exposed as memory, with attention masked so future bars are hidden.

Conditioning supports difficulty, scale type, time signature IDs, and structural control IDs. Stage one is
metadata-conditioned on scale type, time signature, and structural controls by default. Difficulty is supported
but disabled by default until labels are reliable enough to train against. Enabled conditioning IDs are summed
into one prefix vector.
`scale_root` is not currently a model condition and should remain decode metadata for transposition control.

Training can also add an auxiliary validity penalty. The penalty builds a hard-constraint mask from each
teacher-forced prefix and penalizes probability mass assigned to tokens that the generation constraints would
reject from that state. It does not replace the next-token objective. If the ground-truth target is already
invalid for its prefix, that position is excluded from the auxiliary penalty and counted as an invalid-target
metric instead of pushing the model away from the observed token.

## MLflow Metric Naming Protocol

MLflow metric names use slash-separated hierarchy. The aggregation/statistic level must be explicit and must appear
before the measured quantity:

```text
<domain>/<scope>/<statistic>/<metric>
```

Rules:

- `domain` names the subsystem: `dataset`, `model`, or `generation`.
- `scope` names the slice inside that subsystem: examples include `overall`, `diagnostics`, `tokens`,
  `ineligibility`, `train`, `validation`, `soft`, and `hard`.
- `statistic` names how the value should be read in dashboards: `count`, `rate`, or `mean`.
- `metric` is the concrete measured quantity and should not repeat the statistic suffix. Use
  `dataset/diagnostics/mean/right_silence_fraction`, not `dataset/right_silence_fraction_mean`.
- Counts are absolute values, rates and fractions are normalized to `[0, 1]`, and means are arithmetic means over the
  run's relevant samples.
- New MLflow metrics must follow this convention unless they are external tool metrics with fixed names.

Current metric families:

- Dataset processing metrics:
  - `dataset/overall/count/parsed_files`
  - `dataset/overall/count/parsed_successes`
  - `dataset/overall/count/parse_errors`
  - `dataset/overall/rate/parse_success`
  - `dataset/overall/count/segments`
  - `dataset/overall/count/encoded_samples`
  - `dataset/overall/rate/eligible`
  - `dataset/diagnostics/rate/empty_score`
  - `dataset/diagnostics/rate/one_hand_only`
  - `dataset/diagnostics/mean/right_silence_fraction`
  - `dataset/diagnostics/mean/left_silence_fraction`
  - `dataset/diagnostics/mean/both_hands_silence_fraction`
  - `dataset/diagnostics/mean/both_hands_active_fraction`
  - `dataset/diagnostics/mean/hand_activity_balance`
  - `dataset/diagnostics/mean/silent_bar_count`
  - `dataset/diagnostics/mean/silent_bar_fraction`
  - `dataset/diagnostics/mean/silent_edge_bar_count`
  - `dataset/tokens/mean/note_fraction`
  - `dataset/tokens/mean/rest_fraction`
  - `dataset/tokens/mean/hold_fraction`
  - `dataset/scale_match/mean/in_scale_weight_fraction`
  - `dataset/scale_match/mean/out_of_scale_weight_fraction`
  - `dataset/scale_match/mean/best_margin`
  - `dataset/scale_match/mean/observed_pitch_class_count`
  - `dataset/scale_match/mean/tied_best_candidate_count`
  - `dataset/scale_match/rate/declared_match_used`
  - `dataset/scale_match/rate/low_confidence`
  - `dataset/scale_match/rate/ambiguous`
  - `dataset/scale_match/rate/no_pitches`
  - `dataset/ineligibility/count/<reason>`
  - `dataset/ineligibility/rate/<reason>`
- Model training metrics:
  - `model/train/mean/loss`
  - `model/train/mean/perplexity`
  - `model/train/rate/token_accuracy`
  - `model/train/rate/token_kind_accuracy`
  - `model/train/mean/validity_penalty_loss`
  - `model/train/mean/invalid_probability_mass`
  - `model/train/rate/invalid_target`
  - `model/train/mean/<module>_gradient_norm`
  - corresponding `model/validation/...` metrics where validation is available.
- Generation evaluation metrics:
  - `generation/<soft|hard>/count/samples`
  - `generation/<soft|hard>/rate/end`
  - `generation/<soft|hard>/rate/decode_error`
  - `generation/<soft|hard>/rate/constraint_error`
  - `generation/<soft|hard>/rate/constraint_failure`
  - `generation/<soft|hard>/rate/target_bar_completion`
  - `generation/<soft|hard>/mean/constraint_valid_token_fraction`
  - `generation/<soft|hard>/mean/constraint_first_failure_step`
  - `generation/<soft|hard>/mean/bar_count_error`
  - `generation/<soft|hard>/mean/generated_tokens`
  - `generation/<soft|hard>/mean/completed_bars`
  - `generation/<soft|hard>/rate/empty_score`
  - `generation/<soft|hard>/rate/one_hand_only`
  - `generation/<soft|hard>/mean/right_silence_fraction`
  - `generation/<soft|hard>/mean/left_silence_fraction`
  - `generation/<soft|hard>/mean/both_hands_silence_fraction`
  - `generation/<soft|hard>/mean/both_hands_active_fraction`
  - `generation/<soft|hard>/mean/hand_activity_balance`
  - `generation/<soft|hard>/mean/silent_bar_count`
  - `generation/<soft|hard>/mean/silent_bar_fraction`
  - `generation/<soft|hard>/mean/silent_edge_bar_count`
  - `generation/<soft|hard>/mean/note_token_fraction`
  - `generation/<soft|hard>/mean/rest_token_fraction`
  - `generation/<soft|hard>/mean/hold_token_fraction`

## Stage Two Constrained Fine-Tuning

Stage one remains the grammar/vocabulary pretraining phase. It uses the same autoregressive next-token
objective and is not expected to be the final exercise generator by itself. It uses stable metadata
conditioning for scale type and time signature so the model learns controllable meter and mode preferences
from the start. The pretrain default config also enables the auxiliary validity penalty so invalid grammar
choices are discouraged during pretraining without running full constrained decoding inside training.

Stage two is a separate fine-tuning phase. It loads a pretrain checkpoint into the same model shape,
then trains on exercise-style data with conditioning enabled. Until a dedicated exercise-only dataset
exists, stage two can be run on the same dataset as stage one to validate the two-stage pipeline.

Stage two keeps the next-token objective and the same auxiliary validity penalty. Auxiliary heads,
masked-token objectives, and layer freezing are deferred research options, not part of the initial pipeline.
Bar count is a structural control for finetuning and generation requests, but pretraining leaves that control
in the unknown bucket because song segments are not concrete exercise-length examples.

Structural controls are derived automatically from tokenized segments and metadata. They are optional:
each control vocabulary has an explicit unknown/no-control bucket so generation requests can omit a
constraint. Initial bucket thresholds are config-defined. Dataset-quantile buckets may be useful later
for calibration, but are not the source of truth for v1.

Pilot structural controls:

- shortest note duration;
- dotted-note presence;
- maximum notes per same-hand onset;
- maximum notes per same-hand onset under one hand, exposed as a per-hand chord-size control;
- maximum same-hand onset span in semitones;
- maximum same-hand melodic gap;
- static hand placement span;
- target bar count for finetuning and generation;
- scale type and time signature from metadata.

Static hand placement means each hand independently stays within a fixed 5-degree inclusive diatonic
range, computed as `octave_offset * 7 + degree`. Accidentals do not change this placement coordinate.

## Controllable Generation Direction

Generation should use hybrid control:

- soft model conditioning for style and learned preferences;
- hard constrained decoding for exact validity.

The generation request should include:

- `scale_root`
- `scale_type`
- `time_signature`
- `bar_count`
- later controls such as maximum notes per hand, difficulty, and additional style preferences.

Ordinary notes and rests should fit within the remaining duration of the active hand's current bar. Cross-bar
sound should not be modeled by overflowing note durations. Instead, the generator should split the sound at the
barline and emit `HoldToken` at the start of the continuation span.

The decoder state should track:

- active hand;
- current bar;
- per-hand cursor inside the bar;
- previous same-hand note/chord pitch set;
- whether each hand has a continuation available;
- note count per onset and per hand when density controls are active.

The sampler should mask invalid tokens:

- note/rest durations that exceed the active hand's remaining bar time;
- `HoldToken` without a previous same-hand note or chord;
- `HoldToken` durations that exceed the current bar;
- `BarToken` before both hands exactly fill the current measure;
- `EndToken` before the requested number of complete bars;
- same-hand chord notes that would create more than five simultaneous notes for one hand;
- same-hand chord notes that would span more than an octave when onset-span control is active;
- future controls such as too-short durations, disallowed dotted durations, excessive gaps, or too many
  same-hand notes.

Tie probability should be controlled by logits penalties or sampling bias, not by removing tie support.

The first implementation slice lives in `musak_model.generation.constraints`. It is model-agnostic: a
sampling loop can pass the generated prefix token IDs to `allowed_next_token_ids(...)`, then pass the
result to `mask_disallowed_logits(...)` before temperature/top-k/top-p sampling.

The hard state currently tracks:

- active hand;
- completed bar count;
- absolute right- and left-hand cursors;
- previous contiguous same-hand attack end, used for `HoldToken`;
- previous same-hand onset, used for chord joins;
- previous same-hand onset pitches, used for maximum melodic gap;
- pending chord-join state for `JoinWithPreviousToken`.

This lets the mask enforce exact measure length without removing ties. A cross-bar sound is generated
as a note/chord that ends exactly at a barline, followed by a `BarToken`, followed by a same-hand
`HoldToken` in the next bar. Chord notes that start at the end of a bar are allowed to temporarily
overflow only when the next legal token is forced to be `JoinWithPreviousToken`; the join restores the
cursor to the chord onset duration.

Maximum melodic gap is enforced between consecutive same-hand onsets. Large intervals are still allowed
inside a chord: if a note would violate the melodic gap but can join the previous onset, the mask forces
the next token to be `JoinWithPreviousToken`. This keeps chord voicings distinct from melodic leaps.

## Tie and Hold Rules

`HoldToken(duration_id)` means: extend the active hand's previous same-hand note or chord by the specified
duration. It advances only the active hand cursor and creates no new attack.

The parser preserves MusicXML note/chord tie state. The segmenter converts matching same-hand tied
continuations into hold tokens:

- `start` tie events remain normal note/chord attacks and open a same-hand tie state;
- `continue` tie events become `HoldToken` and keep the tie state open;
- `stop` tie events become `HoldToken` and close the tie state;
- partial chord ties and mismatched continuation pitch sets are marked ineligible for training.
- segment windows that start on a tie continuation are marked ineligible, because the hold would not have a
  preceding attack inside the training sample.

Valid examples:

```text
R 1(1:2) L r(1:2) | R h(1:2) L 5(1:4) 6(1:4) |
```

The right hand sustains the previous note across the barline while the left hand plays new attacks in the next bar.

```text
R 1(1:4) 3(1:4) ~ | R h(1:4) |
```

The right-hand chord is extended by the hold. The hold belongs only to the right hand because the active hand is right.

Invalid examples:

```text
R 1(1:4) L h(1:4)
```

The left hand has no previous left-hand note or chord to extend.

```text
R h(1:4)
```

No previous right-hand attack exists.

## Next Implementation Work

The implemented tie slices add:

- `HoldToken` to the tokenizer, text representation, vocabulary, tokenizer snapshot, duration remapping helpers,
  and piano-roll decoding;
- parsed tie state on notes/chords;
- segmenter conversion from tied note/chord continuations to hold tokens;
- ineligibility handling for partial chord ties, mismatched tie continuations, and windows that start on a tie
  continuation.
- Music21 export of held notes/chords as tied fragments split at barlines.
- a model-agnostic hard constraint state for next-token generation masks.
- hard controls for shortest duration, dotted-duration policy, chord size, onset span, and maximum melodic gap.
- finetune structural control extraction, bucketed conditioning IDs, target bar-count conditioning, and fine-tuning
  entrypoint.
- an auxiliary training loss that penalizes probability assigned to hard-invalid next tokens.

Remaining work:

- integrate the generation constraint mask into a model sampling loop;
- add soft logits controls for tie likelihood and other stylistic preferences;
- train stage one and stage two end to end on the full dataset as a pipeline validation pass;
- add a dedicated exercise-only dataset and difficulty controls when available.
