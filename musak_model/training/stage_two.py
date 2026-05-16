from __future__ import annotations

import logging
from pathlib import Path

import torch

from musak_model.conditioning.structural import StructuralControlVocabulary
from musak_model.conditioning.time_signature import TimeSignatureVocabulary
from musak_model.data.config import SegmentationConfig
from musak_model.model import HierarchicalAutoregressiveModel
from musak_model.model.config import ModelConfig
from musak_model.paths import CONDITIONING_CONFIG_PATH
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.checkpoint import load_model_weights
from musak_model.training.config import StageTwoTrainingConfig
from musak_model.training.dataset import build_dataloaders
from musak_model.training.ingestion.config import IngestionConfig
from musak_model.training.ingestion.split import build_split
from musak_model.training.progress import log_split_summary
from musak_model.training.tracking import build_training_tracker
from musak_model.training.trainer import StageOneTrainer, TrainingResult

_LOGGER = logging.getLogger(__name__)


def train_stage_two(
    source_dir: Path,
    *,
    ingestion_config: IngestionConfig,
    segmentation_config: SegmentationConfig,
    training_config: StageTwoTrainingConfig,
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
        conditioning_config_path=conditioning_config_path,
    )
    _LOGGER.info("Model vocabulary size: %s", resolved_model_config.vocabulary_size)
    split = build_split(
        source_dir,
        config=ingestion_config,
        segmentation=segmentation_config,
        tokenization_config=tokenization_config,
        allow_raw_fallback=allow_raw_fallback,
    )
    log_split_summary(split)
    time_signature_vocabulary = TimeSignatureVocabulary(resolved_model_config.conditioning.time_signature)
    structural_control_vocabulary = StructuralControlVocabulary(resolved_model_config.conditioning.structural)
    train_loader, validation_loader = build_dataloaders(
        split,
        batch_size=training_config.batch_size,
        shuffle_train=True,
        num_workers=training_config.num_workers,
        include_conditioning=training_config.use_conditioning,
        include_structural_controls=training_config.use_structural_conditioning,
        time_signature_vocabulary=time_signature_vocabulary,
        token_vocabulary=token_vocabulary,
        structural_control_vocabulary=structural_control_vocabulary,
    )
    model = HierarchicalAutoregressiveModel(resolved_model_config)
    _LOGGER.info("Loading stage-one model weights from: %s", training_config.stage_one_checkpoint)
    load_model_weights(
        training_config.stage_one_checkpoint,
        model=model,
        device=torch.device(training_config.device),
    )
    tracker = build_training_tracker(training_config=training_config)

    with tracker:
        tracker.log_setup(
            training_config=training_config,
            model_config=resolved_model_config,
            split=split,
        )

        trainer = StageOneTrainer(
            model=model,
            config=training_config,
            train_loader=train_loader,
            validation_loader=validation_loader,
            tracker=tracker,
            show_progress=show_progress,
        )

        return trainer.train(invalid_files=split.invalid_files)
