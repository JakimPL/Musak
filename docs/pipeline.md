# Processing and Training Pipeline

This is the user-facing path for the experimental sight-reading model pipeline. The model is not integrated into the
Musak web app yet.

## Dataset Rule

Processing takes a dataset root and writes to `artifacts/processed/<dataset-name>`:

```text
data/pretraining-dataset -> artifacts/processed/pretraining-dataset
data/finetuning-dataset -> artifacts/processed/finetuning-dataset
```

Processing recursively gathers `.mxl`, `.xml`, and `.musicxml` files from the dataset root.

Training takes the same dataset root and looks for matching artifacts under `artifacts/processed/<dataset-name>`.

## Typical End-To-End Run

Process the broad pretraining dataset, process the exercise finetuning dataset, then train both stages:

```bash
DATA_DIR=data/pretraining-dataset make process

DATA_DIR=data/finetuning-dataset \
PROCESS_WHOLE_FILE_SEGMENTS=1 \
PROCESS_DIFFICULTY_LABELS=data/finetuning-difficulty.json \
make process

PRETRAIN_DATA_DIR=data/pretraining-dataset \
FINETUNE_DATA_DIR=data/finetuning-dataset \
FINETUNE_DIFFICULTY_LABELS=data/finetuning-difficulty.json \
DEVICE=cuda \
make train
```

Use `make pretrain` or `make finetune` instead of `make train` when you want to run only one training stage.

## Process A Dataset

Run the complete processing pipeline:

```bash
DATA_DIR=data/pretraining-dataset make process
```

`make process` recursively gathers MusicXML files from `DATA_DIR`, parses compatible two-part piano scores, tokenizes
training examples, computes dataset diagnostics, builds figure-profile artifacts, and logs processing metrics to
MLflow. By default it writes artifacts below `artifacts/processed/<dataset-name>`. No train/validation split exists at this
stage; training creates the split later from the processed dataset.

Useful processing switches:

```bash
DATA_DIR=data/pretraining-dataset make parse
DATA_DIR=data/pretraining-dataset make tokenize
DATA_DIR=data/pretraining-dataset PROCESS_OVERWRITE=1 make process
DATA_DIR=data/pretraining-dataset PROCESS_SKIP_FIGURE_ANALYSIS=1 make process
DATA_DIR=data/finetuning-dataset PROCESS_WHOLE_FILE_SEGMENTS=1 PROCESS_DIFFICULTY_LABELS=data/finetuning-difficulty.json make process
```

`make analyze-n-grams` remains available for standalone figure extraction, but it is not needed after a normal
`make process` run:

```bash
DATA_DIR=data/pretraining-dataset make analyze-n-grams
```

Figure extraction writes resumable batch progress below the encoded run while it is incomplete. Compatible partial
work resumes automatically, and `OVERWRITE=1` restarts from scratch.

Start the local MLflow UI with:

```bash
make mlflow
```

## Train A Model

Run pretraining only:

```bash
PRETRAIN_DATA_DIR=data/pretraining-dataset PRETRAIN_EPOCHS=25 PRETRAIN_DEVICE=cuda make pretrain
```

Run finetuning only from a pretraining checkpoint:

```bash
FINETUNE_DATA_DIR=data/finetuning-dataset \
FINETUNE_DIFFICULTY_LABELS=data/finetuning-difficulty.json \
FINETUNE_EPOCHS=8 \
FINETUNE_DEVICE=cuda \
make finetune
```

Run both training stages:

```bash
PRETRAIN_DATA_DIR=data/pretraining-dataset \
FINETUNE_DATA_DIR=data/finetuning-dataset \
FINETUNE_DIFFICULTY_LABELS=data/finetuning-difficulty.json \
EPOCHS=25 \
DEVICE=cuda \
make train
```

Training logs model metrics to MLflow. Generation evaluation is enabled by the default training configs and logs
sample-quality metrics during training. Training also builds figure profiles for the actual train and validation
partitions and logs `model/split/figure/...` metrics so the split distribution can be inspected. When matching figure
artifacts are available under the processed encoded run, generation evaluation also logs figure comparison metrics;
figure profiles are not generation constraints and do not change sampling.

## Inspect Output

Use the model output explorer notebook to sample from a checkpoint and inspect generated music:

```bash
make notebook-model-output-explorer
```

The explorer can load a reference `figure/all/counts.csv` and display generated figure metrics below the score, player,
and piano roll. These metrics are for comparison only.
