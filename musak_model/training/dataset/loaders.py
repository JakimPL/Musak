import logging
from time import perf_counter
from typing import cast

from torch.utils.data import DataLoader

from musak_model.auxiliary.config import MusicalAuxiliaryTargetConfig
from musak_model.conditioning.structural.vocabulary import StructuralControlVocabulary
from musak_model.conditioning.time_signature import TimeSignatureVocabulary
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.config import TrainingConditioningConfig
from musak_model.training.dataset.collate import collate_training_examples
from musak_model.training.dataset.examples import EncodedExerciseDataset
from musak_model.training.dataset.schema import TrainingBatch
from musak_model.training.ingestion.schema import IngestionSplit

_LOGGER = logging.getLogger(__name__)


def build_dataloaders(
    split: IngestionSplit,
    *,
    batch_size: int,
    shuffle_train: bool,
    num_workers: int,
    time_signature_vocabulary: TimeSignatureVocabulary,
    token_vocabulary: TokenVocabulary,
    musical_auxiliary_targets: MusicalAuxiliaryTargetConfig,
    conditioning: TrainingConditioningConfig,
    structural_control_vocabulary: StructuralControlVocabulary | None = None,
    include_structural_controls: bool = False,
    include_bar_count_control: bool = False,
    max_sequence_length: int | None = None,
) -> tuple[DataLoader[TrainingBatch], DataLoader[TrainingBatch]]:
    _LOGGER.info("Building training dataset examples: samples=%s", len(split.train))
    started_at = perf_counter()
    train_dataset = EncodedExerciseDataset(
        split.train,
        conditioning=conditioning,
        include_structural_controls=include_structural_controls,
        include_bar_count_control=include_bar_count_control,
        time_signature_vocabulary=time_signature_vocabulary,
        token_vocabulary=token_vocabulary,
        musical_auxiliary_targets=musical_auxiliary_targets,
        structural_control_vocabulary=structural_control_vocabulary,
        max_sequence_length=max_sequence_length,
    )
    _LOGGER.info(
        "Built training dataset examples in %.1fs: examples=%s",
        perf_counter() - started_at,
        len(train_dataset),
    )
    _LOGGER.info("Building validation dataset examples: samples=%s", len(split.validation))
    started_at = perf_counter()
    validation_dataset = EncodedExerciseDataset(
        split.validation,
        conditioning=conditioning,
        include_structural_controls=include_structural_controls,
        include_bar_count_control=include_bar_count_control,
        time_signature_vocabulary=time_signature_vocabulary,
        token_vocabulary=token_vocabulary,
        musical_auxiliary_targets=musical_auxiliary_targets,
        structural_control_vocabulary=structural_control_vocabulary,
        max_sequence_length=max_sequence_length,
    )
    _LOGGER.info(
        "Built validation dataset examples in %.1fs: examples=%s",
        perf_counter() - started_at,
        len(validation_dataset),
    )
    _LOGGER.info(
        "Creating DataLoaders: batch_size=%s shuffle_train=%s num_workers=%s",
        batch_size,
        shuffle_train,
        num_workers,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collate_training_examples,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_training_examples,
    )
    return cast(
        tuple[DataLoader[TrainingBatch], DataLoader[TrainingBatch]],
        (
            train_loader,
            validation_loader,
        ),
    )
