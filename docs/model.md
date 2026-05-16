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

Training receives the dataset root and optional processed root. With `data/PDMX` and `processed`, the training code looks for reusable encoded artifacts under `processed/PDMX/encoded/<tokenizer-hash>`. Encoded artifacts are used only when `tokenizer.json` matches the active tokenizer snapshot. Otherwise, training falls back to parsed JSON and then raw MusicXML.

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

## Controllable Generation Direction

Generation should use hybrid control:

- soft model conditioning for style and learned preferences;
- hard constrained decoding for exact validity.

The generation request should include:

- `key_root`
- `scale_type`
- `time_signature`
- `bar_count`
- later controls such as maximum gap, maximum notes per hand, shortest duration, dotted-note policy, and difficulty.

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

Remaining work:

- export held notes/chords with MusicXML tie notation;
- add the constrained sampler that uses the hold rules during generation.
