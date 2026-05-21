import logging
from pathlib import Path
from random import Random
from typing import Final

from music21.exceptions21 import Music21Exception

from musak_model.data.config import SegmentationConfig, SegmentationMode
from musak_model.data.pipeline import process_file, segment_parsed_score
from musak_model.data.schema import Segment
from musak_model.processing.io import load_encoded_jsonl, load_parsed_score_json, load_tokenizer_snapshot_json
from musak_model.processing.manifest import (
    EncodedManifestField,
    ParsedManifestField,
    ParsedManifestStatus,
    read_encoded_manifest,
    read_parsed_manifest,
)
from musak_model.processing.paths import ProcessedDatasetPaths
from musak_model.processing.snapshot import TokenizerSnapshot, build_tokenizer_snapshot
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import BarToken, EndToken, Token
from musak_model.tokens.vocabulary import TokenVocabulary, encode
from musak_model.training.ingestion.config import IngestionConfig
from musak_model.training.ingestion.schema import EncodedExercise, IngestionErrorRecord, IngestionSplit
from musak_shared.files import collect_musicxml_files

_LOGGER = logging.getLogger(__name__)
_FILE_PROCESSING_ERRORS: Final[tuple[type[Exception], ...]] = (
    Music21Exception,
    OSError,
    OverflowError,
    TypeError,
    ValueError,
)


def build_split(
    source_directory: Path,
    *,
    config: IngestionConfig,
    segmentation: SegmentationConfig,
    tokenization_config: TokenizationConfig,
    allow_raw_fallback: bool = True,
) -> IngestionSplit:
    duration_vocabulary = DurationVocabulary(tokenization_config)
    token_vocabulary = TokenVocabulary(duration_vocabulary)

    processed_split = _build_split_from_processed_artifacts(
        source_directory,
        config=config,
        segmentation=segmentation,
        tokenization_config=tokenization_config,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
    )
    if processed_split is not None:
        return processed_split

    if config.processed_root is not None:
        _LOGGER.warning("Processed artifacts were not used; falling back to raw MusicXML from %s", source_directory)

    if not allow_raw_fallback:
        raise ValueError(
            "processed artifacts are unavailable or unusable, and raw MusicXML fallback is disabled. "
            "Pass --data-dir to allow raw fallback or rebuild processed artifacts."
        )

    file_paths = collect_musicxml_files(source_directory)
    validation_files = set(
        _split_validation_files(
            file_paths=file_paths,
            validation_fraction=config.validation_fraction,
            split_seed=config.split_seed,
        )
    )

    train_samples: list[EncodedExercise] = []
    validation_samples: list[EncodedExercise] = []
    invalid_files: list[IngestionErrorRecord] = []

    for file_path in file_paths:
        try:
            segments = process_file(
                file_path,
                duration_vocabulary,
                segmentation_config=segmentation,
                difficulty_labels=config.difficulty_labels,
            )
        except _FILE_PROCESSING_ERRORS as exception:
            invalid_files.append(
                IngestionErrorRecord(
                    file=str(file_path),
                    exception_type=type(exception).__name__,
                    message=str(exception),
                )
            )
            continue

        encoded_samples = _encode_segments(segments, token_vocabulary=token_vocabulary)
        if file_path in validation_files:
            validation_samples.extend(encoded_samples)
        else:
            train_samples.extend(encoded_samples)

    return IngestionSplit(
        train=train_samples,
        validation=validation_samples,
        invalid_files=invalid_files,
    )


def _build_split_from_processed_artifacts(
    source_directory: Path,
    *,
    config: IngestionConfig,
    segmentation: SegmentationConfig,
    tokenization_config: TokenizationConfig,
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
) -> IngestionSplit | None:
    if config.processed_root is None:
        return None

    paths = ProcessedDatasetPaths.from_dataset_root(processed_root=config.processed_root, dataset_root=source_directory)
    parsed_rows = read_parsed_manifest(paths.parsed_manifest_path)
    if not parsed_rows:
        _LOGGER.warning("Processed parsed manifest is missing or empty: %s", paths.parsed_manifest_path)
        return None

    invalid_files = [
        IngestionErrorRecord(
            file=row[ParsedManifestField.SOURCE_PATH],
            exception_type=row[ParsedManifestField.ERROR_TYPE],
            message=row[ParsedManifestField.ERROR_MESSAGE],
        )
        for row in parsed_rows
        if row[ParsedManifestField.STATUS] == ParsedManifestStatus.ERROR.value
    ]
    source_paths = [
        source_directory / row[ParsedManifestField.SOURCE_PATH]
        for row in parsed_rows
        if row[ParsedManifestField.STATUS] == ParsedManifestStatus.SUCCESS.value
    ]
    validation_keys = {
        _source_key(path, source_directory=source_directory)
        for path in _split_validation_files(
            file_paths=source_paths,
            validation_fraction=config.validation_fraction,
            split_seed=config.split_seed,
        )
    }

    snapshot = build_tokenizer_snapshot(
        tokenization_config,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
    )
    encoded_path = paths.encoded_jsonl_path(snapshot.tokenizer_hash)
    if encoded_path.exists() and _encoded_artifacts_match(paths=paths, expected_snapshot=snapshot):
        _validate_encoded_segmentation_mode(
            paths=paths, tokenizer_hash_value=snapshot.tokenizer_hash, segmentation=segmentation
        )
        _LOGGER.info("Using encoded processed artifacts: %s", encoded_path)
        return _split_encoded_samples(
            load_encoded_jsonl(encoded_path),
            validation_keys=validation_keys,
            source_directory=source_directory,
            invalid_files=invalid_files,
        )

    parsed_success_rows = [
        row for row in parsed_rows if row[ParsedManifestField.STATUS] == ParsedManifestStatus.SUCCESS.value
    ]
    if parsed_success_rows:
        _LOGGER.warning("Encoded processed artifacts unavailable; rebuilding training samples from parsed artifacts")
        return _split_from_parsed_scores(
            parsed_success_rows,
            paths=paths,
            source_directory=source_directory,
            validation_keys=validation_keys,
            segmentation=segmentation,
            difficulty_labels=config.difficulty_labels,
            duration_vocabulary=duration_vocabulary,
            token_vocabulary=token_vocabulary,
            invalid_files=invalid_files,
        )

    _LOGGER.warning("Processed parsed manifest has no successful parsed scores: %s", paths.parsed_manifest_path)
    return None


def _encoded_artifacts_match(*, paths: ProcessedDatasetPaths, expected_snapshot: TokenizerSnapshot) -> bool:
    encoded_manifest_path = paths.encoded_manifest_path(expected_snapshot.tokenizer_hash)
    if not encoded_manifest_path.exists():
        _LOGGER.warning("Ignoring encoded artifacts without encoded manifest: %s", encoded_manifest_path)
        return False

    snapshot_path = paths.tokenizer_snapshot_path(expected_snapshot.tokenizer_hash)
    if not snapshot_path.exists():
        _LOGGER.warning("Ignoring encoded artifacts without tokenizer snapshot: %s", snapshot_path)
        return False

    snapshot = load_tokenizer_snapshot_json(snapshot_path)
    if snapshot != expected_snapshot:
        _LOGGER.warning(
            "Ignoring encoded artifacts with tokenizer snapshot mismatch: expected %s, found %s",
            expected_snapshot.tokenizer_hash,
            snapshot.tokenizer_hash,
        )
        return False

    return True


def _validate_encoded_segmentation_mode(
    *,
    paths: ProcessedDatasetPaths,
    tokenizer_hash_value: str,
    segmentation: SegmentationConfig,
) -> None:
    if segmentation.mode != SegmentationMode.WHOLE_FILE:
        return

    encoded_manifest_path = paths.encoded_manifest_path(tokenizer_hash_value)
    rows = read_encoded_manifest(encoded_manifest_path)
    invalid_modes = {
        row.get(EncodedManifestField.SEGMENTATION_MODE, "")
        for row in rows
        if row.get(EncodedManifestField.ENCODED_LINE, "") != ""
        and row.get(EncodedManifestField.SEGMENTATION_MODE, "") != SegmentationMode.WHOLE_FILE.value
    }
    if invalid_modes:
        raise ValueError(
            "finetuning with whole-file segmentation requires encoded artifacts built with "
            f"segmentation_mode={SegmentationMode.WHOLE_FILE.value}; found {sorted(invalid_modes)}"
        )


def _split_encoded_samples(
    samples: list[EncodedExercise],
    *,
    validation_keys: set[str],
    source_directory: Path,
    invalid_files: list[IngestionErrorRecord],
) -> IngestionSplit:
    train_samples: list[EncodedExercise] = []
    validation_samples: list[EncodedExercise] = []
    for sample in samples:
        if _source_key(sample.source_file, source_directory=source_directory) in validation_keys:
            validation_samples.append(sample)
        else:
            train_samples.append(sample)

    return IngestionSplit(
        train=train_samples,
        validation=validation_samples,
        invalid_files=invalid_files,
    )


def _split_from_parsed_scores(
    parsed_rows: list[dict[str, str]],
    *,
    paths: ProcessedDatasetPaths,
    source_directory: Path,
    validation_keys: set[str],
    segmentation: SegmentationConfig,
    difficulty_labels: dict[str, int | None] | None,
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
    invalid_files: list[IngestionErrorRecord],
) -> IngestionSplit:
    train_samples: list[EncodedExercise] = []
    validation_samples: list[EncodedExercise] = []
    for row in parsed_rows:
        relative_source_path = Path(row[ParsedManifestField.SOURCE_PATH])
        source_path = source_directory / relative_source_path
        parsed_path = paths.root / row[ParsedManifestField.PARSED_PATH]
        try:
            score = load_parsed_score_json(parsed_path)
            segments = segment_parsed_score(
                score,
                relative_source_path,
                duration_vocabulary,
                segmentation_config=segmentation,
                difficulty_labels=difficulty_labels,
            )
        except _FILE_PROCESSING_ERRORS as exception:
            invalid_files.append(
                IngestionErrorRecord(
                    file=row[ParsedManifestField.SOURCE_PATH],
                    exception_type=type(exception).__name__,
                    message=str(exception),
                )
            )
            continue

        encoded_samples = _encode_segments(segments, token_vocabulary=token_vocabulary)
        if _source_key(source_path, source_directory=source_directory) in validation_keys:
            validation_samples.extend(encoded_samples)
        else:
            train_samples.extend(encoded_samples)

    return IngestionSplit(
        train=train_samples,
        validation=validation_samples,
        invalid_files=invalid_files,
    )


def _split_validation_files(
    *,
    file_paths: list[Path],
    validation_fraction: float,
    split_seed: int,
) -> list[Path]:
    if not file_paths:
        return []

    shuffled_files = list(file_paths)
    Random(split_seed).shuffle(shuffled_files)
    validation_count = _validation_count(total_files=len(shuffled_files), validation_fraction=validation_fraction)
    return shuffled_files[:validation_count]


def _validation_count(
    *,
    total_files: int,
    validation_fraction: float,
) -> int:
    if total_files <= 1 or validation_fraction <= 0:
        return 0

    count = int(total_files * validation_fraction)
    if count == 0:
        count = 1

    if count >= total_files:
        count = total_files - 1

    return count


def _source_key(path: Path, *, source_directory: Path) -> str:
    try:
        return path.resolve().relative_to(source_directory.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _encode_segments(
    segments: list[Segment],
    *,
    token_vocabulary: TokenVocabulary,
) -> list[EncodedExercise]:
    encoded_samples: list[EncodedExercise] = []
    for segment in segments:
        if not segment.metadata.eligible_for_training:
            continue
        encoded_samples.append(_encode_segment(segment, token_vocabulary=token_vocabulary))

    return encoded_samples


def _encode_segment(
    segment: Segment,
    *,
    token_vocabulary: TokenVocabulary,
) -> EncodedExercise:
    token_ids = encode(segment.tokens, vocabulary=token_vocabulary)
    bar_positions = _build_bar_positions_from_tokens(segment.tokens)
    return EncodedExercise(
        token_ids=token_ids,
        bar_positions=bar_positions,
        hand=None,
        metadata=segment.metadata,
    )


def _build_bar_positions_from_tokens(tokens: list[Token]) -> list[int]:
    bar_index = 0
    bar_positions: list[int] = []

    for token in tokens:
        if isinstance(token, EndToken):
            bar_positions.append(max(bar_index - 1, 0))
            continue

        bar_positions.append(bar_index)
        if isinstance(token, BarToken):
            bar_index += 1

    return bar_positions
