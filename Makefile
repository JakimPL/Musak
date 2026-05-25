.DEFAULT_GOAL := help

PROCESSED_ROOT ?= processed
PROCESS_OVERWRITE ?= $(or $(OVERWRITE),$(OVERWITE))
PROCESS_DISABLE_MLFLOW ?=
PROCESS_MLFLOW_EXPERIMENT ?= musak-process
PROCESS_MLFLOW_RUN_NAME ?=
PROCESS_MLFLOW_TRACKING_URI ?= $(MLFLOW_TRACKING_URI)
PROCESSING_CONFIG ?=
PROCESS_DIFFICULTY_LABELS ?=
PROCESS_WHOLE_FILE_SEGMENTS ?=
PROCESS_PROFILE ?= $(PROFILE)
PROCESS_TOKENIZATION_WORKERS ?=
PROCESS_TOKENIZATION_BATCH_SIZE ?=
PROCESS_SKIP_FIGURE_ANALYSIS ?=
ANALYSIS_CONFIG ?=
ANALYSIS_OUTPUT ?=
ANALYSIS_ENCODED_DIR ?=
ANALYSIS_NO_PROGRESS ?=
APP_HOST ?= 127.0.0.1
APP_PORT ?= 8000
MLFLOW_DIR ?= mlruns
MLFLOW_HOST ?= 127.0.0.1
MLFLOW_PORT ?= 5000
NOTEBOOK_FILES := $(shell grep -l '^[[:space:]]*app[[:space:]]*=[[:space:]]*marimo\.App' notebooks/*.py 2>/dev/null)
NOTEBOOK_NAMES := $(subst _,-,$(basename $(notdir $(NOTEBOOK_FILES))))
NOTEBOOK_TARGETS := $(addprefix notebook-,$(NOTEBOOK_NAMES))
NOTEBOOK_MODE ?= edit

.PHONY: help install test app parse tokenize process analyze-n-grams train pretrain finetune mlflow FORCE

PRETRAIN_DATA_DIR ?= $(DATA_DIR)
PRETRAIN_PROCESSED_DIR ?=
PRETRAIN_EPOCHS ?= $(EPOCHS)
PRETRAIN_DEVICE ?= $(DEVICE)
PRETRAIN_NUM_WORKERS ?= $(NUM_WORKERS)
PRETRAIN_OVERWRITE ?= $(OVERWRITE)
PRETRAIN_CHECKPOINT_DIR ?=
PRETRAIN_RESUME_CHECKPOINT ?= $(if $(PRETRAIN_CHECKPOINT_DIR),$(PRETRAIN_CHECKPOINT_DIR)/latest.pt,checkpoints/pretraining/latest.pt)
PRETRAIN_DIFFICULTY_LABELS ?=
PRETRAIN_WHOLE_FILE_SEGMENTS ?=

FINETUNE_DATA_DIR ?= $(PRETRAIN_DATA_DIR)
FINETUNE_PROCESSED_DIR ?= $(PRETRAIN_PROCESSED_DIR)
FINETUNE_EPOCHS ?= $(EPOCHS)
FINETUNE_DEVICE ?= $(DEVICE)
FINETUNE_NUM_WORKERS ?= $(NUM_WORKERS)
FINETUNE_CHECKPOINT_DIR ?=
FINETUNE_RESUME_CHECKPOINT ?= $(if $(FINETUNE_CHECKPOINT_DIR),$(FINETUNE_CHECKPOINT_DIR)/latest.pt,checkpoints/finetuning/latest.pt)
FINETUNE_DIFFICULTY_LABELS ?=
FINETUNE_WHOLE_FILE_SEGMENTS ?= 1
PRETRAIN_CHECKPOINT ?= checkpoints/pretraining/best.pt

help:
	@printf '%s\n' 'Musak development commands'
	@printf '%s\n' ''
	@printf '%s\n' 'Targets:'
	@printf '%s\n' '  make install          Install Python dev/model dependencies and pre-commit hooks.'
	@printf '%s\n' '  make test             Run the pytest suite used by the pre-push hook.'
	@printf '%s\n' '  make app              Start the Musak FastAPI app with reload enabled.'
	@printf '%s\n' '  make parse            Parse one MusicXML dataset into parsed artifacts.'
	@printf '%s\n' '  make tokenize         Encode parsed artifacts into tokenized dataset artifacts.'
	@printf '%s\n' '  make process          Run parse, tokenize, then figure analysis for one MusicXML dataset.'
	@printf '%s\n' '  make analyze-n-grams  Extract figure n-gram counts from encoded dataset artifacts.'
	@printf '%s\n' '  make pretrain         Train the broad token-distribution pretrain model.'
	@printf '%s\n' '  make finetune         Fine-tune from a pretrain checkpoint with conditioning controls.'
	@printf '%s\n' '  make train            Run pretrain, then finetune.'
	@printf '%s\n' '  make mlflow           Start the local MLflow dashboard.'
	@$(if $(NOTEBOOK_TARGETS),printf '%s\n' '  make notebook-<name>  Start a discovered Marimo notebook.';)
	@$(foreach target,$(NOTEBOOK_TARGETS),printf '%s\n' '    $(target)';)
	@printf '%s\n' ''
	@printf '%s\n' 'Examples:'
	@printf '%s\n' '  make install'
	@printf '%s\n' '  make test'
	@printf '%s\n' '  APP_PORT=8080 make app'
	@printf '%s\n' '  DATA_DIR=data/PDMX PROCESSED_ROOT=processed NUM_WORKERS=8 make process'
	@printf '%s\n' '  DATA_DIR=data/PDMX PROCESSED_ROOT=processed make analyze-n-grams'
	@printf '%s\n' '  DATA_DIR=data/exercises PROCESS_WHOLE_FILE_SEGMENTS=1 PROCESS_DIFFICULTY_LABELS=data/exercises/difficulty_labels.json PROCESS_OVERWRITE=1 make process'
	@printf '%s\n' '  PRETRAIN_PROCESSED_DIR=processed/PDMX PRETRAIN_EPOCHS=25 PRETRAIN_DEVICE=cuda make pretrain'
	@printf '%s\n' '  FINETUNE_PROCESSED_DIR=processed/exercises FINETUNE_DIFFICULTY_LABELS=data/exercises/difficulty_labels.json PRETRAIN_CHECKPOINT=checkpoints/pretraining/best.pt FINETUNE_EPOCHS=8 make finetune'
	@printf '%s\n' '  PRETRAIN_PROCESSED_DIR=processed/PDMX FINETUNE_PROCESSED_DIR=processed/exercises FINETUNE_DIFFICULTY_LABELS=data/exercises/difficulty_labels.json EPOCHS=25 DEVICE=cuda NUM_WORKERS=4 make train'
	@printf '%s\n' '  MLFLOW_DIR=mlruns MLFLOW_PORT=5000 make mlflow'
	@printf '%s\n' ''
	@printf '%s\n' 'Variables:'
	@printf '%s\n' '  APP_HOST              Musak app host. Default: 127.0.0.1'
	@printf '%s\n' '  APP_PORT              Musak app port. Default: 8000'
	@printf '%s\n' '  DATA_DIR              Dataset root for process.'
	@printf '%s\n' '  PROCESSED_ROOT        Processed artifact root for process. Default: processed'
	@printf '%s\n' '  PROCESS_OVERWRITE=1   Pass --overwrite to process. Defaults to OVERWRITE when set.'
	@printf '%s\n' '  PROCESS_DISABLE_MLFLOW=1 disables process MLflow dataset metrics.'
	@printf '%s\n' '  PROCESS_MLFLOW_EXPERIMENT, PROCESS_MLFLOW_RUN_NAME, PROCESS_MLFLOW_TRACKING_URI configure process MLflow logging.'
	@printf '%s\n' '  PROCESSING_CONFIG    Optional parsing/tokenization processing YAML override.'
	@printf '%s\n' '  PROCESS_DIFFICULTY_LABELS Optional difficulty-label JSON/YAML path for process.'
	@printf '%s\n' '  PROCESS_WHOLE_FILE_SEGMENTS=1 passes --whole-file-segments to process.'
	@printf '%s\n' '  PROCESS_TOKENIZATION_WORKERS overrides tokenization worker processes.'
	@printf '%s\n' '  PROCESS_TOKENIZATION_BATCH_SIZE overrides tokenization source files per worker task.'
	@printf '%s\n' '  PROCESS_SKIP_FIGURE_ANALYSIS=1 skips figure analysis during process/tokenize.'
	@printf '%s\n' '  ANALYSIS_CONFIG       Optional figure n-gram analysis YAML override.'
	@printf '%s\n' '  ANALYSIS_OUTPUT       Optional extra figure n-gram CSV output path.'
	@printf '%s\n' '  ANALYSIS_ENCODED_DIR  Optional encoded run directory override when multiple tokenizer runs exist.'
	@printf '%s\n' '  ANALYSIS_NO_PROGRESS=1 disables figure n-gram progress bars.'
	@printf '%s\n' '  PROFILE=1 or PROCESS_PROFILE=1 passes --profile to process.'
	@printf '%s\n' '  MLFLOW_DIR            MLflow tracking directory. Default: mlruns'
	@printf '%s\n' '  MLFLOW_HOST           MLflow dashboard host. Default: 127.0.0.1'
	@printf '%s\n' '  MLFLOW_PORT           MLflow dashboard port. Default: 5000'
	@printf '%s\n' '  PRETRAIN_DATA_DIR     Optional raw pretrain dataset root for raw fallback.'
	@printf '%s\n' '  PRETRAIN_PROCESSED_DIR Dataset-specific pretrain artifacts, e.g. processed/PDMX.'
	@printf '%s\n' '  PRETRAIN_CHECKPOINT_DIR Optional checkpoint output directory override for pretrain.'
	@printf '%s\n' '  PRETRAIN_DIFFICULTY_LABELS Optional difficulty-label JSON/YAML path for pretrain fallback.'
	@printf '%s\n' '  PRETRAIN_WHOLE_FILE_SEGMENTS=1 trains pretrain from whole-file segments.'
	@printf '%s\n' '  FINETUNE_DATA_DIR     Optional raw finetune dataset root for raw fallback. Defaults to PRETRAIN_DATA_DIR.'
	@printf '%s\n' '  FINETUNE_PROCESSED_DIR Dataset-specific finetune artifacts. Defaults to PRETRAIN_PROCESSED_DIR.'
	@printf '%s\n' '  FINETUNE_CHECKPOINT_DIR Optional checkpoint output directory override for finetune.'
	@printf '%s\n' '  FINETUNE_DIFFICULTY_LABELS Required difficulty-label JSON/YAML path for exercise finetuning.'
	@printf '%s\n' '  FINETUNE_WHOLE_FILE_SEGMENTS=1 passes --whole-file-segments to finetune. Default: 1'
	@printf '%s\n' '  PRETRAIN_CHECKPOINT   Checkpoint used by finetune. Default: checkpoints/pretraining/best.pt'
	@printf '%s\n' '  EPOCHS, DEVICE, NUM_WORKERS provide shared defaults.'
	@printf '%s\n' '  OVERWRITE=1 passes --overwrite to process and pretrain checkpoint safety checks.'
	@printf '%s\n' '  RESUME=1 resumes from each stage latest checkpoint and takes precedence over OVERWRITE.'
	@printf '%s\n' '  PRETRAIN_RESUME_CHECKPOINT, FINETUNE_RESUME_CHECKPOINT override resume paths.'
	@printf '%s\n' '  PRETRAIN_EPOCHS, PRETRAIN_DEVICE, PRETRAIN_NUM_WORKERS override pretrain only.'
	@printf '%s\n' '  FINETUNE_EPOCHS, FINETUNE_DEVICE, FINETUNE_NUM_WORKERS override finetune only.'
	@printf '%s\n' '  NOTEBOOK_MODE         Marimo subcommand for notebook targets. Default: edit'

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

parse:
	$(call require_var,DATA_DIR)
	$(call process_dataset_command,parse)

tokenize:
	$(call require_var,DATA_DIR)
	$(call process_dataset_command,tokenize)

process:
	$(call require_var,DATA_DIR)
	$(call process_dataset_command,process)

analyze-n-grams:
	$(call require_var,DATA_DIR)
	$(call analyze_n_grams_command)

train:
	$(MAKE) pretrain
	$(MAKE) finetune

pretrain:
	$(call require_training_source,PRETRAIN_DATA_DIR,PRETRAIN_PROCESSED_DIR)
	uv run python scripts/pretrain.py \
		$(call optional_arg,PRETRAIN_DATA_DIR,--data-dir) \
		$(call optional_arg,PRETRAIN_PROCESSED_DIR,--processed-dir) \
		$(call optional_arg,PRETRAIN_CHECKPOINT_DIR,--checkpoint-dir) \
		$(call optional_arg,PRETRAIN_EPOCHS,--epochs) \
		$(call optional_arg,PRETRAIN_DEVICE,--device) \
		$(call optional_arg,PRETRAIN_NUM_WORKERS,--num-workers) \
		$(call optional_arg,PRETRAIN_DIFFICULTY_LABELS,--difficulty-labels) \
		$(call optional_flag,PRETRAIN_WHOLE_FILE_SEGMENTS,--whole-file-segments) \
		$(call optional_resume_checkpoint,PRETRAIN_RESUME_CHECKPOINT) \
		$(call optional_non_resume_flag,PRETRAIN_OVERWRITE,--overwrite)

finetune:
	$(call require_training_source,FINETUNE_DATA_DIR,FINETUNE_PROCESSED_DIR)
	$(call require_var,FINETUNE_DIFFICULTY_LABELS)
	uv run python scripts/finetune.py \
		$(call optional_arg,FINETUNE_DATA_DIR,--data-dir) \
		$(call optional_arg,FINETUNE_PROCESSED_DIR,--processed-dir) \
		$(call optional_arg,FINETUNE_CHECKPOINT_DIR,--checkpoint-dir) \
		--pretrain-checkpoint "$(PRETRAIN_CHECKPOINT)" \
		$(call optional_arg,FINETUNE_EPOCHS,--epochs) \
		$(call optional_arg,FINETUNE_DEVICE,--device) \
		$(call optional_arg,FINETUNE_NUM_WORKERS,--num-workers) \
		--difficulty-labels "$(FINETUNE_DIFFICULTY_LABELS)" \
		$(call optional_flag,FINETUNE_WHOLE_FILE_SEGMENTS,--whole-file-segments) \
		$(call optional_resume_checkpoint,FINETUNE_RESUME_CHECKPOINT)

mlflow:
	uv run mlflow ui \
		--backend-store-uri "file:$(MLFLOW_DIR)" \
		--host "$(MLFLOW_HOST)" \
		--port "$(MLFLOW_PORT)"

notebook-%: FORCE
	$(call require_notebook,$*)
	uv run marimo "$(NOTEBOOK_MODE)" "$(call notebook_file,$*)"

FORCE:

define require_var
	$(if $($(1)),,$(error $(1) is required))
endef

define require_training_source
	$(if $(or $($(1)),$($(2))),,$(error either $(1) or $(2) is required))
endef

define require_notebook
	$(if $(filter $(call notebook_file,$(1)),$(NOTEBOOK_FILES)),,$(error notebook-$(1) does not match a Marimo notebook in notebooks/))
endef

define notebook_file
notebooks/$(subst -,_,$(1)).py
endef

define optional_arg
	$(if $($(1)),$(2) "$($(1))",)
endef

define optional_flag
	$(if $($(1)),$(2),)
endef

define optional_non_resume_flag
	$(if $(RESUME),,$(call optional_flag,$(1),$(2)))
endef

define optional_resume_checkpoint
	$(if $(RESUME),--resume-checkpoint "$($(1))",)
endef

define analyze_n_grams_command
	uv run python scripts/extract_figures.py \
		--data-dir "$(DATA_DIR)" \
		--processed-root "$(PROCESSED_ROOT)" \
		$(call optional_arg,ANALYSIS_CONFIG,--analysis-config) \
		$(call optional_arg,ANALYSIS_OUTPUT,--output) \
		$(call optional_arg,ANALYSIS_ENCODED_DIR,--encoded-dir) \
		$(call optional_flag,ANALYSIS_NO_PROGRESS,--no-progress)
endef

define process_dataset_command
	uv run python scripts/process_dataset.py \
		--data-dir "$(DATA_DIR)" \
		--processed-dir "$(PROCESSED_ROOT)" \
		--stage "$(1)" \
		$(call optional_arg,PROCESSING_CONFIG,--processing-config) \
		$(call optional_arg,NUM_WORKERS,--workers) \
		$(call optional_arg,PROCESS_TOKENIZATION_WORKERS,--tokenization-workers) \
		$(call optional_arg,PROCESS_TOKENIZATION_BATCH_SIZE,--tokenization-batch-size) \
		$(call optional_arg,PROCESS_DIFFICULTY_LABELS,--difficulty-labels) \
		$(call optional_flag,PROCESS_WHOLE_FILE_SEGMENTS,--whole-file-segments) \
		$(call optional_arg,ANALYSIS_CONFIG,--analysis-config) \
		$(call optional_arg,ANALYSIS_OUTPUT,--analysis-output) \
		$(call optional_flag,ANALYSIS_NO_PROGRESS,--no-progress) \
		$(call optional_flag,PROCESS_SKIP_FIGURE_ANALYSIS,--skip-figure-analysis) \
		$(call optional_flag,PROCESS_PROFILE,--profile) \
		$(call optional_flag,PROCESS_OVERWRITE,--overwrite) \
		$(call optional_flag,PROCESS_DISABLE_MLFLOW,--disable-mlflow) \
		--mlflow-experiment-name "$(PROCESS_MLFLOW_EXPERIMENT)" \
		$(call optional_arg,PROCESS_MLFLOW_RUN_NAME,--mlflow-run-name) \
		$(call optional_arg,PROCESS_MLFLOW_TRACKING_URI,--mlflow-tracking-uri)
endef
