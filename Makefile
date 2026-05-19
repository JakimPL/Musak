.DEFAULT_GOAL := help

.PHONY: help install test app process train pretrain finetune mlflow

PROCESSED_ROOT ?= processed
PROCESS_STAGE ?= all
APP_HOST ?= 127.0.0.1
APP_PORT ?= 8000
MLFLOW_DIR ?= mlruns
MLFLOW_HOST ?= 127.0.0.1
MLFLOW_PORT ?= 5000

PRETRAIN_DATA_DIR ?= $(DATA_DIR)
PRETRAIN_PROCESSED_DIR ?=
PRETRAIN_EPOCHS ?= $(EPOCHS)
PRETRAIN_DEVICE ?= $(DEVICE)
PRETRAIN_NUM_WORKERS ?= $(NUM_WORKERS)
PRETRAIN_OVERWRITE ?= $(OVERWRITE)

FINETUNE_DATA_DIR ?= $(PRETRAIN_DATA_DIR)
FINETUNE_PROCESSED_DIR ?= $(PRETRAIN_PROCESSED_DIR)
FINETUNE_EPOCHS ?= $(EPOCHS)
FINETUNE_DEVICE ?= $(DEVICE)
FINETUNE_NUM_WORKERS ?= $(NUM_WORKERS)
PRETRAIN_CHECKPOINT ?= checkpoints/pretraining/best.pt

help:
	@printf '%s\n' 'Musak development commands'
	@printf '%s\n' ''
	@printf '%s\n' 'Targets:'
	@printf '%s\n' '  make install          Install Python dev/model dependencies and pre-commit hooks.'
	@printf '%s\n' '  make test             Run the pytest suite used by the pre-push hook.'
	@printf '%s\n' '  make app              Start the Musak FastAPI app with reload enabled.'
	@printf '%s\n' '  make process          Parse and encode one MusicXML dataset.'
	@printf '%s\n' '  make pretrain   Train the broad token-distribution pretrain model.'
	@printf '%s\n' '  make finetune   Fine-tune from a pretrain checkpoint with conditioning controls.'
	@printf '%s\n' '  make train            Run pretrain, then finetune.'
	@printf '%s\n' '  make mlflow           Start the local MLflow dashboard.'
	@printf '%s\n' ''
	@printf '%s\n' 'Examples:'
	@printf '%s\n' '  make install'
	@printf '%s\n' '  make test'
	@printf '%s\n' '  APP_PORT=8080 make app'
	@printf '%s\n' '  DATA_DIR=data/PDMX PROCESSED_ROOT=processed NUM_WORKERS=8 make process'
	@printf '%s\n' '  PRETRAIN_DATA_DIR=data/PDMX PRETRAIN_PROCESSED_DIR=processed/PDMX PRETRAIN_EPOCHS=25 PRETRAIN_DEVICE=cuda make pretrain'
	@printf '%s\n' '  FINETUNE_DATA_DIR=data/Exercises FINETUNE_PROCESSED_DIR=processed/Exercises PRETRAIN_CHECKPOINT=checkpoints/pretraining/best.pt FINETUNE_EPOCHS=8 make finetune'
	@printf '%s\n' '  PRETRAIN_DATA_DIR=data/PDMX PRETRAIN_PROCESSED_DIR=processed/PDMX FINETUNE_DATA_DIR=data/Exercises FINETUNE_PROCESSED_DIR=processed/Exercises EPOCHS=25 DEVICE=cuda NUM_WORKERS=4 make train'
	@printf '%s\n' '  MLFLOW_DIR=mlruns MLFLOW_PORT=5000 make mlflow'
	@printf '%s\n' ''
	@printf '%s\n' 'Variables:'
	@printf '%s\n' '  APP_HOST              Musak app host. Default: 127.0.0.1'
	@printf '%s\n' '  APP_PORT              Musak app port. Default: 8000'
	@printf '%s\n' '  DATA_DIR              Dataset root for process.'
	@printf '%s\n' '  PROCESSED_ROOT        Processed artifact root for process. Default: processed'
	@printf '%s\n' '  PROCESS_STAGE         parsed, encoded, or all. Default: all'
	@printf '%s\n' '  MLFLOW_DIR            MLflow tracking directory. Default: mlruns'
	@printf '%s\n' '  MLFLOW_HOST           MLflow dashboard host. Default: 127.0.0.1'
	@printf '%s\n' '  MLFLOW_PORT           MLflow dashboard port. Default: 5000'
	@printf '%s\n' '  PRETRAIN_DATA_DIR     Raw pretrain dataset root.'
	@printf '%s\n' '  PRETRAIN_PROCESSED_DIR Dataset-specific pretrain artifacts, e.g. processed/PDMX.'
	@printf '%s\n' '  FINETUNE_DATA_DIR     Raw finetune dataset root. Defaults to PRETRAIN_DATA_DIR.'
	@printf '%s\n' '  FINETUNE_PROCESSED_DIR Dataset-specific finetune artifacts. Defaults to PRETRAIN_PROCESSED_DIR.'
	@printf '%s\n' '  PRETRAIN_CHECKPOINT   Checkpoint used by finetune. Default: checkpoints/pretraining/best.pt'
	@printf '%s\n' '  EPOCHS, DEVICE, NUM_WORKERS provide shared defaults.'
	@printf '%s\n' '  OVERWRITE=1 passes --overwrite to pretrain checkpoint safety checks.'
	@printf '%s\n' '  PRETRAIN_EPOCHS, PRETRAIN_DEVICE, PRETRAIN_NUM_WORKERS override pretrain only.'
	@printf '%s\n' '  FINETUNE_EPOCHS, FINETUNE_DEVICE, FINETUNE_NUM_WORKERS override finetune only.'

install:
	uv sync --extra dev --group model
	uv run pre-commit install

test:
	uv run pytest tests

app:
	uv run uvicorn musak.api.main:app \
		--reload \
		--host "$(APP_HOST)" \
		--port "$(APP_PORT)"

process:
	$(call require_var,DATA_DIR)
	uv run python scripts/process_dataset.py \
		--data-dir "$(DATA_DIR)" \
		--processed-dir "$(PROCESSED_ROOT)" \
		--stage "$(PROCESS_STAGE)" \
		$(call optional_arg,NUM_WORKERS,--workers)

train: pretrain finetune

pretrain:
	$(call require_var,PRETRAIN_DATA_DIR)
	$(call require_var,PRETRAIN_PROCESSED_DIR)
	uv run python scripts/pretrain.py \
		--data-dir "$(PRETRAIN_DATA_DIR)" \
		--processed-dir "$(PRETRAIN_PROCESSED_DIR)" \
		$(call optional_arg,PRETRAIN_EPOCHS,--epochs) \
		$(call optional_arg,PRETRAIN_DEVICE,--device) \
		$(call optional_arg,PRETRAIN_NUM_WORKERS,--num-workers) \
		$(call optional_flag,PRETRAIN_OVERWRITE,--overwrite)

finetune:
	$(call require_var,FINETUNE_DATA_DIR)
	$(call require_var,FINETUNE_PROCESSED_DIR)
	uv run python scripts/finetune.py \
		--data-dir "$(FINETUNE_DATA_DIR)" \
		--processed-dir "$(FINETUNE_PROCESSED_DIR)" \
		--pretrain-checkpoint "$(PRETRAIN_CHECKPOINT)" \
		$(call optional_arg,FINETUNE_EPOCHS,--epochs) \
		$(call optional_arg,FINETUNE_DEVICE,--device) \
		$(call optional_arg,FINETUNE_NUM_WORKERS,--num-workers)

mlflow:
	uv run mlflow ui \
		--backend-store-uri "file:$(MLFLOW_DIR)" \
		--host "$(MLFLOW_HOST)" \
		--port "$(MLFLOW_PORT)"

define require_var
	$(if $($(1)),,$(error $(1) is required))
endef

define optional_arg
	$(if $($(1)),$(2) "$($(1))",)
endef

define optional_flag
	$(if $($(1)),$(2),)
endef
