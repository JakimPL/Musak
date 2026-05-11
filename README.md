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
