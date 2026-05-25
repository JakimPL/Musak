# Musak

Musak is a web application for musical exercises. The current app includes interval, chord inversion, and rhythm
practice pages backed by generated notation and audio.

The repository also contains an experimental sight-reading model pipeline. That pipeline can process MusicXML datasets,
train autoregressive models, and inspect generated outputs, but it is not integrated into the Musak web app yet.

## Run The Web App

After installation, start the local FastAPI app:

```bash
./run.sh
```

Then open `http://localhost:8000`.

Use a custom port with:

```bash
./run.sh 9000
```

Enable verbose browser-facing errors during development with:

```bash
DEBUG=1 ./run.sh
```

The main app routes are:

- `/intervals/` for interval exercises.
- `/inversions/` for chord inversion exercises.
- `/rhythm/` for rhythm exercises.

## Install

Run the installer to set up system dependencies, Python dependencies, and the default soundfont:

```bash
./install.sh
```

This installs `lilypond`, `fluidsynth`, and `ffmpeg` via the system package manager. On Debian/Ubuntu, it also links
the system `FluidR3_GM.sf2` soundfont to `soundfont/st_concert.sf2`.

On macOS or unsupported systems, install those packages manually and place a `.sf2` soundfont at
`soundfont/st_concert.sf2`. A freely available option is GeneralUser GS.

Python dependencies are managed with `uv`. To sync the app and development dependencies manually:

```bash
uv sync --extra dev
```

To include the model pipeline, notebooks, and training dependencies:

```bash
uv sync --extra dev --group model
```

## Development Commands

Common entrypoints are exposed through `make`:

```bash
make app
make test
make mlflow
make notebook-model-output-explorer
```

Run `make help` for the full list of app, processing, training, MLflow, and notebook commands.

## Model Pipeline

The model pipeline is separate from the web application. It is current work for a future sight-reading generator and is
not used by the exercise pages yet.

`make process` prepares a MusicXML dataset for training. It recursively gathers `.mxl`, `.xml`, and `.musicxml` files
under `DATA_DIR`, parses compatible two-part piano scores, tokenizes training examples, computes dataset diagnostics,
builds figure-profile artifacts, and logs processing metrics to MLflow. By default, reusable artifacts are written
under `processed/<dataset-name>`.

Process a broad pretraining dataset:

```bash
DATA_DIR=data/PDMX make process
```

Process an exercise-style finetuning dataset with whole-file segments and difficulty labels:

```bash
DATA_DIR=data/exercises \
PROCESS_WHOLE_FILE_SEGMENTS=1 \
PROCESS_DIFFICULTY_LABELS=data/exercises/difficulty_labels.json \
make process
```

Train pretraining only:

```bash
PRETRAIN_DATA_DIR=data/PDMX make pretrain
```

Train finetuning only from a pretraining checkpoint:

```bash
FINETUNE_DATA_DIR=data/exercises \
FINETUNE_DIFFICULTY_LABELS=data/exercises/difficulty_labels.json \
PRETRAIN_CHECKPOINT=checkpoints/pretraining/best.pt \
make finetune
```

Run both model stages:

```bash
PRETRAIN_DATA_DIR=data/PDMX \
FINETUNE_DATA_DIR=data/exercises \
FINETUNE_DIFFICULTY_LABELS=data/exercises/difficulty_labels.json \
make train
```

See [docs/pipeline.md](docs/pipeline.md) for the fuller processing and training workflow.

## Inspect Model Outputs

Start MLflow to review processing and training metrics:

```bash
make mlflow
```

Open the model output explorer notebook to sample from a checkpoint and inspect generated notation, piano-roll playback,
and figure metrics:

```bash
make notebook-model-output-explorer
```

Figure metrics in the explorer are for comparison only. They do not constrain generation.

## Tests

```bash
uv run pytest tests/
```

## Technical References

- [docs/pipeline.md](docs/pipeline.md): model dataset processing and training commands.
- [docs/model.md](docs/model.md): token semantics, MusicXML processing assumptions, generation evaluation, and model
  internals.
- [docs/metrics.md](docs/metrics.md): dataset and generation metric families.
