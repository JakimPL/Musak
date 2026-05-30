from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from musak_model.data.config import SegmentationConfig
from musak_model.data.pipeline import segment_parsed_score
from musak_model.data.schema import Segment
from musak_model.decoder import encoded_exercise_to_segment
from musak_model.processing.config import ProcessingConfig
from musak_model.processing.io import load_encoded_jsonl, load_parsed_score_json, load_tokenizer_snapshot_json
from musak_model.processing.manifest import EncodedManifestField
from musak_model.processing.paths import ENCODED_JSONL_NAME, TOKENIZER_SNAPSHOT_NAME, ProcessedDatasetPaths
from musak_model.processing.snapshot import TokenizerSnapshot
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.schema import EncodedExercise


@dataclass(frozen=True)
class EncodedShard:
    path: Path
    samples: list[EncodedExercise]
    snapshot: TokenizerSnapshot
    duration_vocabulary: DurationVocabulary
    token_vocabulary: TokenVocabulary


@dataclass(frozen=True)
class EncodedManifestSelection:
    row: Mapping[str, object]
    duration_vocabulary: DurationVocabulary
    token_vocabulary: TokenVocabulary
    segment: Segment
    shard: EncodedShard | None = None
    encoded_line: int | None = None


def load_encoded_shard(path: Path) -> EncodedShard:
    snapshot_path = path.parent / TOKENIZER_SNAPSHOT_NAME
    snapshot = load_tokenizer_snapshot_json(snapshot_path)
    tokenization_config = TokenizationConfig.model_validate(snapshot.tokenization_config)
    duration_vocabulary = DurationVocabulary(tokenization_config)
    token_vocabulary = TokenVocabulary(duration_vocabulary)
    return EncodedShard(
        path=path,
        samples=load_encoded_jsonl(path),
        snapshot=snapshot,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
    )


def encoded_sample_to_segment(
    sample: EncodedExercise,
    *,
    shard: EncodedShard,
) -> Segment:
    return encoded_exercise_to_segment(sample, token_vocabulary=shard.token_vocabulary)


def load_encoded_manifest_selection(
    row: Mapping[str, object],
    *,
    dataset_dir: Path,
    encoded_directory: Path | None = None,
) -> EncodedManifestSelection:
    if (encoded_line := _optional_encoded_line(row)) is not None:
        encoded_shard_path = _encoded_shard_path(
            row, dataset_directory=dataset_dir, encoded_directory=encoded_directory
        )
        shard = load_encoded_shard(encoded_shard_path)
        if encoded_line >= len(shard.samples):
            raise IndexError(f"encoded line {encoded_line} is outside shard with {len(shard.samples)} sample(s)")

        segment = encoded_sample_to_segment(shard.samples[encoded_line], shard=shard)
        return EncodedManifestSelection(
            row=row,
            duration_vocabulary=shard.duration_vocabulary,
            token_vocabulary=shard.token_vocabulary,
            segment=segment,
            shard=shard,
            encoded_line=encoded_line,
        )

    duration_vocabulary, token_vocabulary = _vocabularies_from_encoded_directory(encoded_directory)
    segment = _segment_from_manifest_row(
        row,
        dataset_dir=dataset_dir,
        duration_vocabulary=duration_vocabulary,
    )
    return EncodedManifestSelection(
        row=row,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
        segment=segment,
    )


def load_encoded_manifest_selections(
    rows: Sequence[Mapping[str, object]],
    *,
    dataset_dir: Path,
    encoded_directory: Path | None = None,
) -> list[EncodedManifestSelection]:
    shard_cache: dict[Path, EncodedShard] = {}
    selections: list[EncodedManifestSelection] = []
    fallback_vocabularies: tuple[DurationVocabulary, TokenVocabulary] | None = None
    for row in rows:
        if (encoded_line := _optional_encoded_line(row)) is not None:
            encoded_shard_path = _encoded_shard_path(
                row,
                dataset_directory=dataset_dir,
                encoded_directory=encoded_directory,
            )
            shard = shard_cache.get(encoded_shard_path)
            if shard is None:
                shard = load_encoded_shard(encoded_shard_path)
                shard_cache[encoded_shard_path] = shard
            if encoded_line >= len(shard.samples):
                raise IndexError(f"encoded line {encoded_line} is outside shard with {len(shard.samples)} sample(s)")

            selections.append(
                EncodedManifestSelection(
                    row=row,
                    duration_vocabulary=shard.duration_vocabulary,
                    token_vocabulary=shard.token_vocabulary,
                    segment=encoded_sample_to_segment(shard.samples[encoded_line], shard=shard),
                    shard=shard,
                    encoded_line=encoded_line,
                )
            )
            continue

        if fallback_vocabularies is None:
            fallback_vocabularies = _vocabularies_from_encoded_directory(encoded_directory)
        duration_vocabulary, token_vocabulary = fallback_vocabularies
        selections.append(
            EncodedManifestSelection(
                row=row,
                duration_vocabulary=duration_vocabulary,
                token_vocabulary=token_vocabulary,
                segment=_segment_from_manifest_row(
                    row,
                    dataset_dir=dataset_dir,
                    duration_vocabulary=duration_vocabulary,
                ),
            )
        )

    return selections


def _vocabularies_from_encoded_directory(encoded_directory: Path | None) -> tuple[DurationVocabulary, TokenVocabulary]:
    if encoded_directory is None:
        raise ValueError("select an encoded run before previewing ineligible rows")

    snapshot = load_tokenizer_snapshot_json(encoded_directory / TOKENIZER_SNAPSHOT_NAME)
    tokenization_config = TokenizationConfig.model_validate(snapshot.tokenization_config)
    duration_vocabulary = DurationVocabulary(tokenization_config)
    return duration_vocabulary, TokenVocabulary(duration_vocabulary)


def _segment_from_manifest_row(
    row: Mapping[str, object],
    *,
    dataset_dir: Path,
    duration_vocabulary: DurationVocabulary,
) -> Segment:
    parsed_path = _parsed_path(row, dataset_directory=dataset_dir)
    window_start_bar = _integer_field(row, EncodedManifestField.WINDOW_START_BAR)
    bar_count = _integer_field(row, EncodedManifestField.BAR_COUNT)
    score = load_parsed_score_json(parsed_path)
    segments = segment_parsed_score(
        score,
        Path(_string_field(row, EncodedManifestField.SOURCE_PATH, default=parsed_path.name)),
        duration_vocabulary,
        segmentation_config=SegmentationConfig(window_bars=bar_count, stride_bars=1),
        scale_matcher_config=ProcessingConfig.load().tokenization.scale_matcher,
    )
    for segment in segments:
        if segment.metadata.window_start_bar == window_start_bar:
            return segment

    raise ValueError(f"no segment found at window_start_bar={window_start_bar}, bar_count={bar_count}")


def _parsed_path(
    row: Mapping[str, object],
    *,
    dataset_directory: Path,
) -> Path:
    value = row.get(str(EncodedManifestField.PARSED_PATH), row.get(EncodedManifestField.PARSED_PATH))
    if not _is_missing(value):
        return dataset_directory / str(value)

    source_id = row.get(str(EncodedManifestField.SOURCE_ID), row.get(EncodedManifestField.SOURCE_ID))
    if not _is_missing(source_id):
        return ProcessedDatasetPaths(root=dataset_directory).parsed_score_path(str(source_id))

    raise ValueError("selected manifest row has no parsed_path or source_id")


def default_encoded_browser_root(
    processed_root: Path,
    dataset_root: Path,
) -> Path:
    paths = ProcessedDatasetPaths.from_dataset_root(processed_root=processed_root, dataset_root=dataset_root)
    return paths.root / "encoded"


def _encoded_shard_path(
    row: Mapping[str, object],
    *,
    dataset_directory: Path,
    encoded_directory: Path | None,
) -> Path:
    value = row.get(str(EncodedManifestField.ENCODED_SHARD), row.get(EncodedManifestField.ENCODED_SHARD))
    if _is_missing(value) and encoded_directory is not None:
        return encoded_directory / ENCODED_JSONL_NAME

    if not isinstance(value, str) or value == "":
        raise ValueError("selected manifest row has no encoded shard")

    return dataset_directory / value


def _optional_encoded_line(row: Mapping[str, object]) -> int | None:
    value = row.get(str(EncodedManifestField.ENCODED_LINE), row.get(EncodedManifestField.ENCODED_LINE))
    if _is_missing(value):
        return None

    if not isinstance(value, (int, float, str)):
        raise ValueError(f"encoded line must be numeric, got {type(value).__name__}")

    return int(value)


def _integer_field(row: Mapping[str, object], field: EncodedManifestField) -> int:
    value = row.get(str(field), row.get(field))
    if _is_missing(value):
        raise ValueError(f"selected manifest row has no {field.value}")
    if not isinstance(value, (int, float, str)):
        raise ValueError(f"{field.value} must be numeric, got {type(value).__name__}")

    return int(value)


def _string_field(
    row: Mapping[str, object],
    field: EncodedManifestField,
    *,
    default: str | None = None,
) -> str:
    value = row.get(str(field), row.get(field))
    if _is_missing(value):
        if default is not None:
            return default

        raise ValueError(f"selected manifest row has no {field.value}")
    if not isinstance(value, str):
        return str(value)

    return value


def _is_missing(value: object) -> bool:
    return value is None or value == "" or value != value
