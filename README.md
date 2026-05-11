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

Requires Python 3.11+. Install the project and its dependencies:

```bash
pip install -e ".[dev]"
```

## Running

```bash
uvicorn api.main:app --reload
```

Then open [http://localhost:8000](http://localhost:8000).

To enable verbose error messages in the browser:

```bash
DEBUG=1 uvicorn api.main:app --reload
```

## Tests

```bash
pytest tests/
```
