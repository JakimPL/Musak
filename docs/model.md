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
- `metadata`: key root, scale type, time signature, source file, segment window, optional difficulty label, and extracted difficulty features.

The current encoded PDMX artifact is a unified two-hand stream. It is not split into separate right-hand and left-hand samples.

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

Key root is external metadata. Because pitches are stored as scale degrees, the same token stream can be decoded in different roots by changing `metadata.key_root`.

Scale type is also metadata, but it affects pitch-class decoding. The tokenizer has multiple scale types, while current usable data is dominated by major-key material.

`JoinWithPreviousToken` is not a tie token. It represents simultaneous note onsets, usually chord notes. True prolongation is represented by `HoldToken`.

## Current Training Logic

`StageOneTrainer` trains an autoregressive next-token model with teacher forcing:

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

Conditioning currently supports difficulty, scale type, and time signature IDs. These are summed into one prefix vector. `key_root` is not currently a model condition and should remain decode metadata for transposition control.

## Stage Two Constrained Fine-Tuning

Stage one remains the grammar/vocabulary pretraining phase. It uses the same autoregressive next-token
objective and is not expected to be the final exercise generator by itself.

Stage two is a separate fine-tuning phase. It loads a stage-one checkpoint into the same model shape,
then trains on exercise-style data with conditioning enabled. Until a dedicated exercise-only dataset
exists, stage two can be run on the same dataset as stage one to validate the two-stage pipeline.

Stage two keeps the next-token objective. Auxiliary heads, masked-token objectives, and layer freezing
are deferred research options, not part of the initial pipeline.

Structural controls are derived automatically from tokenized segments and metadata. They are optional:
each control vocabulary has an explicit unknown/no-control bucket so generation requests can omit a
constraint. Initial bucket thresholds are config-defined. Dataset-quantile buckets may be useful later
for calibration, but are not the source of truth for v1.

Pilot structural controls:

- shortest note duration;
- dotted-note presence;
- maximum notes per same-hand onset;
- maximum same-hand melodic gap;
- static hand placement span;
- scale type, time signature, and bar count from metadata.

Static hand placement means each hand independently stays within a fixed 5-degree inclusive diatonic
range, computed as `octave_offset * 7 + degree`. Accidentals do not change this placement coordinate.

## Controllable Generation Direction

Generation should use hybrid control:

- soft model conditioning for style and learned preferences;
- hard constrained decoding for exact validity.

The generation request should include:

- `key_root`
- `scale_type`
- `time_signature`
- `bar_count`
- later controls such as maximum notes per hand, difficulty, and additional style preferences.

Ordinary notes and rests should fit within the remaining duration of the active hand's current bar. Cross-bar sound should not be modeled by overflowing note durations. Instead, the generator should split the sound at the barline and emit `HoldToken` at the start of the continuation span.

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
- future controls such as too-short durations, disallowed dotted durations, excessive gaps, or too many same-hand notes.

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

`HoldToken(duration_id)` means: extend the active hand's previous same-hand note or chord by the specified duration. It advances only the active hand cursor and creates no new attack.

The parser preserves MusicXML note/chord tie state. The segmenter converts matching same-hand tied continuations into hold tokens:

- `start` tie events remain normal note/chord attacks and open a same-hand tie state;
- `continue` tie events become `HoldToken` and keep the tie state open;
- `stop` tie events become `HoldToken` and close the tie state;
- partial chord ties and mismatched continuation pitch sets are marked ineligible for training.
- segment windows that start on a tie continuation are marked ineligible, because the hold would not have a preceding attack inside the training sample.

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

- `HoldToken` to the tokenizer, text representation, vocabulary, tokenizer snapshot, duration remapping helpers, and piano-roll decoding;
- parsed tie state on notes/chords;
- segmenter conversion from tied note/chord continuations to hold tokens;
- ineligibility handling for partial chord ties, mismatched tie continuations, and windows that start on a tie continuation.
- Music21 export of held notes/chords as tied fragments split at barlines.
- a model-agnostic hard constraint state for next-token generation masks.
- hard controls for shortest duration, dotted-duration policy, chord size, and maximum melodic gap.
- stage-two structural control extraction, bucketed conditioning IDs, and fine-tuning entrypoint.

Remaining work:

- integrate the generation constraint mask into a model sampling loop;
- add soft logits controls for tie likelihood and other stylistic preferences;
- train stage one and stage two end to end on the full dataset as a pipeline validation pass;
- add a dedicated exercise-only dataset and difficulty controls when available.
