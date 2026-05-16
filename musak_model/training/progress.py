from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Final, TypeVar, cast

from tqdm.auto import tqdm

from musak_model.training.ingestion.schema import EncodedExercise, IngestionSplit

_LOGGER = logging.getLogger(__name__)
_MAX_LOGGED_SOURCE_FILES: Final[int] = 20
_T = TypeVar("_T")


def progress(
    values: Iterable[_T],
    *,
    description: str,
    unit: str,
    enabled: bool,
    total: int | None = None,
) -> Iterable[_T]:
    if not enabled:
        return values

    return cast(Iterable[_T], tqdm(values, total=total, desc=description, unit=unit))


def log_split_summary(split: IngestionSplit) -> None:
    train_sources = _source_files(split.train)
    validation_sources = _source_files(split.validation)
    ingested_sources = tuple(sorted(set(train_sources) | set(validation_sources)))
    _LOGGER.info("Training samples: %s", len(split.train))
    _LOGGER.info("Validation samples: %s", len(split.validation))
    _LOGGER.info("Training source files: %s", len(train_sources))
    _LOGGER.info("Validation source files: %s", len(validation_sources))
    _LOGGER.info("Ingested source files: %s", len(ingested_sources))
    _LOGGER.info("Invalid source files: %s", len(split.invalid_files))
    _log_ingested_sources(ingested_sources)


def _source_files(samples: list[EncodedExercise]) -> tuple[str, ...]:
    return tuple(sorted({_source_path(sample.source_file) for sample in samples}))


def _source_path(source_file: Path) -> str:
    return source_file.as_posix()


def _log_ingested_sources(source_files: tuple[str, ...]) -> None:
    if not source_files:
        return

    for source_file in source_files[:_MAX_LOGGED_SOURCE_FILES]:
        _LOGGER.info("Ingested file: %s", source_file)

    if len(source_files) > _MAX_LOGGED_SOURCE_FILES:
        _LOGGER.info(
            "Ingested file list truncated after %s entries; run with --log-level DEBUG for the full list",
            _MAX_LOGGED_SOURCE_FILES,
        )
        for source_file in source_files[_MAX_LOGGED_SOURCE_FILES:]:
            _LOGGER.debug("Ingested file: %s", source_file)
