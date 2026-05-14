from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile

from music21 import converter
from music21.exceptions21 import Music21Exception
from music21.metadata import Metadata
from music21.stream.base import Score

from musak_model.common.files import collect_musicxml_files
from musak_model.data.config import SegmentationConfig
from musak_model.data.parser import parse_score
from musak_model.data.pipeline import segment_parsed_score
from musak_model.data.schema import ParsedScore
from musak_model.processing.ids import source_id
from musak_model.processing.io import append_jsonl, load_parsed_score_json, write_json_model
from musak_model.processing.manifest import (
    EncodedManifestField,
    ParsedManifestField,
    ParsedManifestStatus,
    encoded_row,
    parsed_error_row,
    parsed_success_row,
    read_encoded_manifest,
    write_encoded_manifest,
    write_parsed_manifest,
)
from musak_model.processing.paths import ProcessedDatasetPaths
from musak_model.processing.snapshot import TokenizerSnapshot, build_tokenizer_snapshot
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.split import _encode_segment

type ProcessingStage = Literal["parsed", "encoded", "all"]

_PROCESSING_ERRORS: tuple[type[Exception], ...] = (
    Music21Exception,
    OSError,
    ParseError,
    BadZipFile,
    TypeError,
    ValueError,
)

_TITLE_EXTRACTION_ERRORS: tuple[type[Exception], ...] = (
    Music21Exception,
    OSError,
    ParseError,
    BadZipFile,
)


@dataclass(frozen=True)
class ProcessDatasetResult:
    parsed_manifest_path: Path
    encoded_manifest_path: Path | None
    tokenizer_snapshot_path: Path | None
    parsed_count: int
    encoded_count: int
    error_count: int


def process_dataset(
    dataset_root: Path,
    *,
    processed_root: Path,
    dataset_name: str,
    segmentation: SegmentationConfig,
    tokenization_config: TokenizationConfig,
    stage: ProcessingStage = "all",
    difficulty_labels: dict[str, int] | None = None,
    overwrite: bool = False,
) -> ProcessDatasetResult:
    paths = ProcessedDatasetPaths.from_roots(processed_root=processed_root, dataset_name=dataset_name)
    duration_vocabulary = DurationVocabulary(tokenization_config)
    token_vocabulary = TokenVocabulary(duration_vocabulary)
    snapshot = build_tokenizer_snapshot(
        tokenization_config,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
    )

    parsed_rows, parsed_scores = _process_parsed_scores(
        dataset_root,
        paths=paths,
        overwrite=overwrite,
    )
    write_parsed_manifest(parsed_rows, paths.parsed_manifest_path)

    encoded_manifest_path: Path | None = None
    tokenizer_snapshot_path: Path | None = None
    encoded_count = 0
    if stage in {"encoded", "all"}:
        encoded_count, encoded_manifest_path, tokenizer_snapshot_path = _process_encoded_segments(
            parsed_scores,
            dataset_root=dataset_root,
            paths=paths,
            snapshot=snapshot,
            segmentation=segmentation,
            duration_vocabulary=duration_vocabulary,
            token_vocabulary=token_vocabulary,
            difficulty_labels=difficulty_labels,
            overwrite=overwrite,
        )

    return ProcessDatasetResult(
        parsed_manifest_path=paths.parsed_manifest_path,
        encoded_manifest_path=encoded_manifest_path,
        tokenizer_snapshot_path=tokenizer_snapshot_path,
        parsed_count=sum(
            1 for row in parsed_rows if row[ParsedManifestField.STATUS] == ParsedManifestStatus.SUCCESS.value
        ),
        encoded_count=encoded_count,
        error_count=sum(
            1 for row in parsed_rows if row[ParsedManifestField.STATUS] == ParsedManifestStatus.ERROR.value
        ),
    )


def _process_parsed_scores(
    dataset_root: Path,
    *,
    paths: ProcessedDatasetPaths,
    overwrite: bool,
) -> tuple[list[dict[str, object]], list[tuple[str, Path, Path, ParsedScore]]]:
    rows: list[dict[str, object]] = []
    parsed_scores: list[tuple[str, Path, Path, ParsedScore]] = []
    for source_path in collect_musicxml_files(dataset_root):
        source_id_value = source_id(source_path, dataset_root=dataset_root)
        parsed_path = paths.parsed_score_path(source_id_value)
        title = _score_title(source_path)
        try:
            if parsed_path.exists() and not overwrite:
                score = load_parsed_score_json(parsed_path)
            else:
                score = parse_score(source_path)
                write_json_model(score, parsed_path, overwrite=overwrite)
        except _PROCESSING_ERRORS as exception:
            rows.append(
                parsed_error_row(
                    source_id_value=source_id_value,
                    source_path=source_path,
                    dataset_root=dataset_root,
                    title=title,
                    exception=exception,
                )
            )
            continue

        rows.append(
            parsed_success_row(
                source_id_value=source_id_value,
                source_path=source_path,
                dataset_root=dataset_root,
                title=title,
                parsed_path=parsed_path,
                processed_root=paths.root,
                score=score,
            )
        )
        parsed_scores.append((source_id_value, source_path, parsed_path, score))

    return rows, parsed_scores


def _process_encoded_segments(
    parsed_scores: list[tuple[str, Path, Path, ParsedScore]],
    *,
    dataset_root: Path,
    paths: ProcessedDatasetPaths,
    snapshot: TokenizerSnapshot,
    segmentation: SegmentationConfig,
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
    difficulty_labels: dict[str, int] | None,
    overwrite: bool,
) -> tuple[int, Path, Path]:
    encoded_jsonl_path = paths.encoded_jsonl_path(snapshot.tokenizer_hash)
    encoded_manifest_path = paths.encoded_manifest_path(snapshot.tokenizer_hash)
    tokenizer_snapshot_path = paths.tokenizer_snapshot_path(snapshot.tokenizer_hash)
    if (
        not overwrite
        and encoded_jsonl_path.exists()
        and encoded_manifest_path.exists()
        and tokenizer_snapshot_path.exists()
    ):
        encoded_rows = read_encoded_manifest(encoded_manifest_path)
        encoded_count = sum(1 for row in encoded_rows if row[EncodedManifestField.ENCODED_LINE] != "")
        return encoded_count, encoded_manifest_path, tokenizer_snapshot_path

    if overwrite and encoded_jsonl_path.exists():
        encoded_jsonl_path.unlink()

    write_json_model(snapshot, tokenizer_snapshot_path, overwrite=True)
    rows: list[dict[str, object]] = []
    encoded_count = 0
    for source_id_value, source_path, parsed_path, score in parsed_scores:
        segments = segment_parsed_score(
            score,
            source_path,
            segmentation=segmentation,
            difficulty_labels=difficulty_labels,
            duration_vocabulary=duration_vocabulary,
        )
        for segment in segments:
            encoded_sample = (
                _encode_segment(segment, token_vocabulary=token_vocabulary)
                if segment.metadata.eligible_for_training
                else None
            )
            encoded_line = append_jsonl(encoded_sample, encoded_jsonl_path) if encoded_sample is not None else None
            if encoded_sample is not None:
                encoded_count += 1
            rows.append(
                encoded_row(
                    source_id_value=source_id_value,
                    source_path=source_path,
                    dataset_root=dataset_root,
                    parsed_path=parsed_path,
                    processed_root=paths.root,
                    segment=segment,
                    encoded_sample=encoded_sample,
                    encoded_shard=encoded_jsonl_path,
                    encoded_line=encoded_line,
                )
            )

    write_encoded_manifest(rows, encoded_manifest_path)
    return encoded_count, encoded_manifest_path, tokenizer_snapshot_path


def _score_title(path: Path) -> str:
    try:
        raw = converter.parse(str(path))
    except _TITLE_EXTRACTION_ERRORS:
        return ""

    if not isinstance(raw, Score):
        return ""

    metadata = raw.metadata
    if not isinstance(metadata, Metadata):
        return ""

    title = metadata.title
    if not isinstance(title, str):
        return ""

    return title
