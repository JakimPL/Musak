# Musak

A set of web tools for ear training.

## Installation

### External dependencies

Run the provided script to install system packages and set up the soundfont:

```bash
./install.sh
```

This installs `lilypond`, `fluidsynth`, and `ffmpeg` via the system package manager and, on Debian/Ubuntu, symlinks the system soundfont (`FluidR3_GM.sf2`) to `soundfont/st_concert.sf2`.

On macOS or unsupported systems, install those packages manually and place a `.sf2` soundfont at `soundfont/st_concert.sf2`. A freely available option is [GeneralUser GS](https://schristiancollins.com/generaluser.php).

### Python environment

Requires [uv](https://docs.astral.sh/uv/). Python 3.13 is managed automatically. `./install.sh` runs `uv sync` for you, but you can also run it manually:

```bash
uv sync --extra dev
```

## Model Token Text Format

Token sequences use a compact text format for logging, debugging, and tests. The text form is a representation of tokens only; scale, key, and other segment metadata remain external context.

Tokens are separated by spaces. The canonical grammar is:

```text
Hand:      R | L
Note:      DEGREE ACCIDENTAL? OCTAVE? DURATION
Rest:      r DURATION
Hold:      h DURATION
Join:      ~
Start:     BOS
Bar:       |
End:       ‖

DEGREE:    1..7
ACCIDENTAL: ♯ | ♭
OCTAVE:    ↑N | ↓N        # omitted means 0
DURATION:  (NUM:DEN)
```

Examples:

```text
R 6♯↑1(1:4) 3(1:4) ~ L r(1:8) 1↓1(1:8) | ‖
```

`6♯↑1(1:4)` means scale degree 6, raised by one semitone, octave offset +1, with duration 1/4. `~` joins the preceding token to the previous note onset, so `n` simultaneous notes are represented with `n - 1` join tokens in the unified stream. This can join notes across `R` and `L` when they share an onset; decoding keeps independent time cursors for each hand.

`~` joins a note to the previous onset for chord representation. `h(NUM:DEN)` extends the active hand's previous same-hand note or chord by the given duration; it is used for tied/held continuations rather than new attacks.

`BOS` is the learned beginning-of-sequence token. Encoded dataset artifacts store musical tokens only and end with `‖`; training prepends `BOS` to the model input so the first musical token is learned as `BOS -> first_token`. Generation should seed the model with the tokenizer vocabulary's `start_token_id`.

Text serialization must use true duration fractions, not duration vocabulary IDs. Parsing text back into tokens converts `(NUM:DEN)` through the active duration vocabulary and should raise a tokenizer-specific error when the duration is unsupported.

## MusicXML Piano Part Policy

The parser accepts only two-part piano scores for the sight-reading model. A score must contain exactly two parts. Parts with no explicit instrument are accepted as piano-compatible; parts with an explicit piano MIDI program are accepted; parts with an explicit non-piano instrument are rejected.

Right and left hand assignment is based on pitch center, not part order or part name. The part with the higher median MIDI pitch is the right hand, and the other part is the left hand. Scores with an empty pitched part or identical pitch centers are rejected as ambiguous.

## Dataset Processing and Training

For the full user-facing workflow, see [docs/pipeline.md](docs/pipeline.md).

Dataset roots and processed artifacts follow one directory rule:

```text
<data-dir> -> <processed-dir>/<data-dir.name>
```

For example, processing `data/PDMX` with the default processed root writes to `processed/PDMX`. Pass the dataset root (`data/PDMX`), not an internal folder such as `data/PDMX/mxl`.

```bash
DATA_DIR=data/PDMX PROCESSED_ROOT=processed NUM_WORKERS=8 make process
```

`make process` runs parsing, tokenization, dataset evaluation diagnostics, and figure-profile extraction in one
processing MLflow run. The processed layout is:

```text
processed/PDMX/
  parsed.csv
  parsed/<first_source_hash_char>/<source_hash>.json
  encoded/<tokenizer_hash>/
    tokenizer.json
    encoded.csv
    data-00000.jsonl
    figure/
      config.yml
      all/
        counts.csv
        profile.json
      by_sample.jsonl
```

Training can work from encoded JSONL, parsed JSON, or raw MusicXML. `--processed-dir` is the processed artifact directory with the dataset name. `--data-dir` is optional when processed artifacts are usable, and is required only when raw MusicXML fallback is needed.

```bash
PRETRAIN_PROCESSED_DIR=processed/PDMX PRETRAIN_EPOCHS=25 PRETRAIN_DEVICE=cuda make pretrain
```

Finetuning is a separate stage initialized from a pretraining checkpoint:

```bash
FINETUNE_PROCESSED_DIR=processed/exercises \
FINETUNE_DIFFICULTY_LABELS=data/exercises/difficulty_labels.json \
PRETRAIN_CHECKPOINT=checkpoints/pretraining/best.pt \
FINETUNE_EPOCHS=8 \
make finetune
```

Run both stages with:

```bash
PRETRAIN_PROCESSED_DIR=processed/PDMX \
FINETUNE_PROCESSED_DIR=processed/exercises \
FINETUNE_DIFFICULTY_LABELS=data/exercises/difficulty_labels.json \
DEVICE=cuda \
make train
```

To allow raw fallback, pass both directories:

```bash
PRETRAIN_DATA_DIR=data/PDMX PRETRAIN_PROCESSED_DIR=processed/PDMX make pretrain
```

When both directories are supplied, their dataset names must match. For example, `--data-dir data/PDMX --processed-dir processed/PDMX` is valid; `--data-dir data/PDMX --processed-dir processed` is not a training path. Processing still takes a processed root and writes the dataset subdirectory, while training takes the resolved processed dataset directory.

When `--processed-dir` is provided, training first looks for encoded artifacts under `processed/PDMX/encoded/<tokenizer_hash>`. Encoded artifacts are reused only when `tokenizer.json` matches the active tokenization config. If matching encoded data is unavailable, training falls back to parsed artifacts. If no usable processed artifacts exist, training parses raw MusicXML only when `--data-dir` was supplied; otherwise it exits with an error.

Parsed artifacts can be re-tokenized with a different tokenization config. Encoded artifacts cannot; they are already tokenized and are selected by tokenizer hash. Paths stored in manifests are relative to the dataset or processed artifact directory and are informational, not a replacement for passing `--data-dir` and `--processed-dir`.

## Running

```bash
./run.sh
```

Or with a custom port:

```bash
./run.sh 9000
```

Then open [http://localhost:8000](http://localhost:8000).

To enable verbose error messages in the browser:

```bash
DEBUG=1 ./run.sh
```

## Tests

```bash
uv run pytest tests/
```
