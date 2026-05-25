# Processing and Training Pipeline

This is the user-facing path for turning a MusicXML dataset into processed artifacts, then training the model in one or
two stages.

## Dataset Rule

Processing takes a raw dataset root and writes under a processed root:

```text
data/PDMX -> processed/PDMX
data/exercises -> processed/exercises
```

Pass the dataset root, not an internal folder such as `data/PDMX/mxl`.

Training takes the dataset-specific processed directory, for example `processed/PDMX`. A raw `data/...` directory is
optional during training and is only needed when you want MusicXML fallback if processed artifacts are missing or stale.

## Typical End-To-End Run

Process the broad pretraining dataset, process the exercise finetuning dataset, then train both stages:

```bash
DATA_DIR=data/PDMX PROCESSED_ROOT=processed NUM_WORKERS=8 make process

DATA_DIR=data/exercises \
PROCESSED_ROOT=processed \
PROCESS_WHOLE_FILE_SEGMENTS=1 \
PROCESS_DIFFICULTY_LABELS=data/exercises/difficulty_labels.json \
NUM_WORKERS=8 \
make process

PRETRAIN_PROCESSED_DIR=processed/PDMX \
FINETUNE_PROCESSED_DIR=processed/exercises \
FINETUNE_DIFFICULTY_LABELS=data/exercises/difficulty_labels.json \
DEVICE=cuda \
NUM_WORKERS=4 \
make train
```

Use `make pretrain` or `make finetune` instead of `make train` when you want to run only one training stage.

## Process A Dataset

Run the complete processing pipeline:

```bash
DATA_DIR=data/PDMX PROCESSED_ROOT=processed NUM_WORKERS=8 make process
```

`make process` runs parsing, tokenization, dataset evaluation diagnostics, and figure-profile extraction in one MLflow
run. It writes:

```text
processed/PDMX/
  parsed.csv
  parsed/<source-hash-prefix>/<source-hash>.json
  encoded/<tokenizer-hash>/
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

Useful processing switches:

```bash
DATA_DIR=data/PDMX make parse
DATA_DIR=data/PDMX make tokenize
DATA_DIR=data/PDMX PROCESS_OVERWRITE=1 make process
DATA_DIR=data/PDMX PROCESS_SKIP_FIGURE_ANALYSIS=1 make process
DATA_DIR=data/exercises PROCESS_WHOLE_FILE_SEGMENTS=1 PROCESS_DIFFICULTY_LABELS=data/exercises/difficulty_labels.json make process
```

`make analyze-n-grams` remains available for standalone figure extraction, but it is not needed after a normal
`make process` run:

```bash
DATA_DIR=data/PDMX PROCESSED_ROOT=processed make analyze-n-grams
```

Start the local MLflow UI with:

```bash
make mlflow
```

## Train A Model

Run pretraining only:

```bash
PRETRAIN_PROCESSED_DIR=processed/PDMX PRETRAIN_EPOCHS=25 PRETRAIN_DEVICE=cuda make pretrain
```

Run finetuning only from a pretraining checkpoint:

```bash
FINETUNE_PROCESSED_DIR=processed/exercises \
FINETUNE_DIFFICULTY_LABELS=data/exercises/difficulty_labels.json \
PRETRAIN_CHECKPOINT=checkpoints/pretraining/best.pt \
FINETUNE_EPOCHS=8 \
FINETUNE_DEVICE=cuda \
make finetune
```

Run both training stages:

```bash
PRETRAIN_PROCESSED_DIR=processed/PDMX \
FINETUNE_PROCESSED_DIR=processed/exercises \
FINETUNE_DIFFICULTY_LABELS=data/exercises/difficulty_labels.json \
EPOCHS=25 \
DEVICE=cuda \
NUM_WORKERS=4 \
make train
```

Add raw dataset directories only when you want raw MusicXML fallback:

```bash
PRETRAIN_DATA_DIR=data/PDMX PRETRAIN_PROCESSED_DIR=processed/PDMX make pretrain
FINETUNE_DATA_DIR=data/exercises FINETUNE_PROCESSED_DIR=processed/exercises FINETUNE_DIFFICULTY_LABELS=data/exercises/difficulty_labels.json make finetune
```

Training logs model metrics to MLflow. Generation evaluation is enabled by the default training configs and logs
sample-quality metrics during training. When matching figure artifacts are available under the processed encoded run,
generation evaluation also logs figure comparison metrics; figure profiles are not generation constraints and do not
change sampling.

## Inspect Output

Use the model output explorer notebook to sample from a checkpoint and inspect generated music:

```bash
make notebook-model-output-explorer
```

The explorer can load a reference `figure/all/counts.csv` and display generated figure metrics below the score, player,
and piano roll. These metrics are for comparison only.
