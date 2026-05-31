from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter

import torch

from musak_model.conditioning.structural.vocabulary import StructuralControlVocabulary
from musak_model.conditioning.time_signature import TimeSignatureVocabulary
from musak_model.data.config import SegmentationConfig
from musak_model.evaluation import GenerationSuiteEvaluator
from musak_model.model import HierarchicalAutoregressiveModel
from musak_model.model.config import ModelConfig
from musak_model.paths import CONDITIONING_CONFIG_PATH
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.checkpoint import load_model_weights
from musak_model.training.config import FinetuningTrainingConfig
from musak_model.training.dataset.loaders import build_dataloaders
from musak_model.training.ingestion.config import IngestionConfig
from musak_model.training.ingestion.split import build_split
from musak_model.training.metrics import build_token_attribute_lookup, build_token_kind_ids
from musak_model.training.progress import log_split_summary
from musak_model.training.stages.auxiliary_profiles import split_musical_auxiliary_profile_metrics
from musak_model.training.stages.figure_profiles import (
    load_generation_figure_profile_artifacts,
    split_figure_profile_metrics,
)
from musak_model.training.stages.pretraining import PretrainingTrainer, TrainingResult
from musak_model.training.tracking import build_training_tracker
from musak_model.training.validity import TrainingValidityMaskBuilder

_LOGGER = logging.getLogger(__name__)


def finetune(
    source_directory: Path,
    *,
    ingestion_config: IngestionConfig,
    segmentation_config: SegmentationConfig,
    training_config: FinetuningTrainingConfig,
    tokenization_config: TokenizationConfig,
    model_config: ModelConfig | None = None,
    conditioning_config_path: Path = CONDITIONING_CONFIG_PATH,
    show_progress: bool = False,
    allow_raw_fallback: bool = True,
) -> TrainingResult:
    _LOGGER.info("Building train/validation split")
    duration_vocabulary = DurationVocabulary(tokenization_config)
    token_vocabulary = TokenVocabulary(duration_vocabulary)
    resolved_model_config = model_config or ModelConfig.load(
        vocabulary_size=token_vocabulary.vocabulary_size,
        duration_vocabulary_size=token_vocabulary.duration_vocabulary.vocabulary_size(),
        output_mode=training_config.event_objective.mode,
        musical_auxiliary_targets=training_config.musical_auxiliary_targets,
        conditioning_config_path=conditioning_config_path,
    )
    _LOGGER.info("Model vocabulary size: %s", resolved_model_config.vocabulary_size)
    split = build_split(
        source_directory,
        config=ingestion_config,
        segmentation=segmentation_config,
        tokenization_config=tokenization_config,
        allow_raw_fallback=allow_raw_fallback,
        show_progress=show_progress,
    )
    log_split_summary(split)
    time_signature_vocabulary = TimeSignatureVocabulary(resolved_model_config.conditioning.time_signature)
    structural_control_vocabulary = StructuralControlVocabulary(resolved_model_config.conditioning.structural)
    _LOGGER.info("Building train/validation DataLoaders")
    started_at = perf_counter()
    train_loader, validation_loader = build_dataloaders(
        split,
        batch_size=training_config.optimization.batch_size,
        shuffle_train=True,
        num_workers=training_config.runtime.num_workers,
        conditioning=training_config.conditioning,
        include_structural_controls=training_config.conditioning.use_structural_conditioning,
        include_bar_count_control=training_config.conditioning.use_structural_conditioning,
        time_signature_vocabulary=time_signature_vocabulary,
        token_vocabulary=token_vocabulary,
        musical_auxiliary_targets=resolved_model_config.musical_auxiliary_targets,
        structural_control_vocabulary=structural_control_vocabulary,
        max_sequence_length=resolved_model_config.transformer.max_sequence_length,
    )
    _LOGGER.info("Built train/validation DataLoaders in %.1fs", perf_counter() - started_at)
    figure_profile_artifacts = (
        load_generation_figure_profile_artifacts(
            source_directory=source_directory,
            ingestion_config=ingestion_config,
            tokenization_config=tokenization_config,
        )
        if training_config.generation_evaluation.enabled
        else None
    )
    _LOGGER.info("Initializing model")
    started_at = perf_counter()
    model = HierarchicalAutoregressiveModel(resolved_model_config)
    _LOGGER.info("Initialized model in %.1fs", perf_counter() - started_at)
    _LOGGER.info("Loading pretrain model weights from: %s", training_config.checkpoints.pretraining_checkpoint)
    started_at = perf_counter()
    load_model_weights(
        training_config.checkpoints.pretraining_checkpoint,
        model=model,
        device=torch.device(training_config.runtime.device),
    )
    _LOGGER.info("Loaded pretrain model weights in %.1fs", perf_counter() - started_at)
    tracker = build_training_tracker(training_config=training_config)

    with tracker:
        tracker.log_setup(
            training_config=training_config,
            model_config=resolved_model_config,
            split=split,
        )
        tracker.log_split_figure_metrics(
            metrics={
                **split_figure_profile_metrics(
                    split,
                    token_vocabulary=token_vocabulary,
                    tokenization_config=tokenization_config,
                    workers=training_config.runtime.num_workers,
                    show_progress=show_progress,
                ),
                **split_musical_auxiliary_profile_metrics(
                    split,
                    token_vocabulary=token_vocabulary,
                    target_config=resolved_model_config.musical_auxiliary_targets,
                ),
            }
        )

        trainer = PretrainingTrainer(
            model=model,
            config=training_config,
            train_loader=train_loader,
            validation_loader=validation_loader,
            tracker=tracker,
            show_progress=show_progress,
            token_kind_ids=build_token_kind_ids(token_vocabulary),
            token_attribute_lookup=build_token_attribute_lookup(token_vocabulary),
            validity_mask_builder=TrainingValidityMaskBuilder(token_vocabulary),
            generation_evaluator=GenerationSuiteEvaluator(
                config=training_config.generation_evaluation,
                conditioning=training_config.conditioning,
                model_config=resolved_model_config,
                token_vocabulary=token_vocabulary,
                duration_vocabulary=duration_vocabulary,
                include_bar_count_control=True,
                figure_profile_artifacts=figure_profile_artifacts,
            ),
        )

        return trainer.train(invalid_files=split.invalid_files)
