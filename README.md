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
Join:      ~
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

`6♯↑1(1:4)` means scale degree 6, raised by one semitone, octave offset +1, with duration 1/4. `~` joins the preceding token to the previous note onset, so `n` simultaneous notes are represented with `n - 1` join tokens in the unified stream. This can join notes across `R` and `L` because the unified stream has one musical time cursor.

Text serialization must use true duration fractions, not duration vocabulary IDs. Parsing text back into tokens converts `(NUM:DEN)` through the active duration vocabulary and should raise a tokenizer-specific error when the duration is unsupported.

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
