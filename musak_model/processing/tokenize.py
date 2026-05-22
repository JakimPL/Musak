from __future__ import annotations

import csv
import hashlib
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeVar, cast

from pydantic import BaseModel
from tqdm.auto import tqdm

from musak_model.data.config import (
    DEFAULT_SCALE_MATCH_MAXIMUM_EXPLANATION_PITCH_CLASS_COUNT,
    DEFAULT_SCALE_MATCH_MAXIMUM_UNEXPLAINED_WEIGHT_FRACTION,
    DEFAULT_SCALE_MATCH_SELECTION_SCORE_MARGIN,
    DEFAULT_SCALE_MATCH_SUPPORT_SCORE_MARGIN,
    DataProcessingConfig,
    SegmentationConfig,
)
from musak_model.data.pipeline import segment_parsed_score
from musak_model.data.schema import Segment, SegmentIneligibilityReason
from musak_model.evaluation.diagnostics import diagnose_segment
from musak_model.processing.io import write_json_model
from musak_model.processing.manifest import (
    ENCODED_MANIFEST_FIELDS,
    EncodedManifestField,
    encoded_row,
    read_encoded_manifest,
)
from musak_model.processing.parse import ParsedScoreArtifact, load_parsed_score_artifacts
from musak_model.processing.paths import ProcessedDatasetPaths
from musak_model.processing.profiler import NULL_PROCESSING_PROFILER, ProcessingProfilerProtocol
from musak_model.processing.snapshot import TokenizerSnapshot, build_tokenizer_snapshot
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.split import _encode_segment

_T = TypeVar("_T")
_LOGGER = logging.getLogger(__name__)

_TOKENIZATION_STATE_VERSION: Final[int] = 1
_TOKENIZATION_STATE_HEADER: Final[str] = "header"
_TOKENIZATION_SOURCE_COMPLETED: Final[str] = "source_completed"


@dataclass(frozen=True)
class TokenizeDatasetResult:
    parsed_manifest_path: Path
    encoded_manifest_path: Path
    tokenizer_snapshot_path: Path
    encoded_count: int
    segment_count: int
    scale_match_support_score_margin: float = DEFAULT_SCALE_MATCH_SUPPORT_SCORE_MARGIN
    scale_match_selection_score_margin: float = DEFAULT_SCALE_MATCH_SELECTION_SCORE_MARGIN
    scale_match_maximum_unexplained_weight_fraction: float = DEFAULT_SCALE_MATCH_MAXIMUM_UNEXPLAINED_WEIGHT_FRACTION
    scale_match_maximum_explanation_pitch_class_count: int = DEFAULT_SCALE_MATCH_MAXIMUM_EXPLANATION_PITCH_CLASS_COUNT


@dataclass(frozen=True)
class _DifficultyLabelStats:
    labeled: int
    explicit_unlabeled: int
    unspecified: int


@dataclass(frozen=True)
class _TokenizationResumeState:
    completed_source_ids: frozenset[str]
    encoded_line_count: int
    manifest_row_count: int
    encoded_count: int
    state_key_matches: bool


def tokenize_dataset(
    dataset_root: Path,
    *,
    processed_root: Path,
    segmentation_config: SegmentationConfig,
    tokenization_config: TokenizationConfig,
    data_processing_config: DataProcessingConfig,
    difficulty_labels: dict[str, int | None] | None,
    overwrite: bool,
    show_progress: bool,
    profiler: ProcessingProfilerProtocol = NULL_PROCESSING_PROFILER,
) -> TokenizeDatasetResult:
    paths = ProcessedDatasetPaths.from_dataset_root(processed_root=processed_root, dataset_root=dataset_root)
    parsed_scores = load_parsed_score_artifacts(dataset_root, processed_root=processed_root)
    duration_vocabulary = DurationVocabulary(tokenization_config)
    token_vocabulary = TokenVocabulary(duration_vocabulary)
    snapshot = build_tokenizer_snapshot(
        tokenization_config,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
    )
    return tokenize_parsed_scores(
        parsed_scores,
        dataset_root=dataset_root,
        paths=paths,
        snapshot=snapshot,
        segmentation_config=segmentation_config,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
        difficulty_labels=difficulty_labels,
        data_processing_config=data_processing_config,
        overwrite=overwrite,
        show_progress=show_progress,
        profiler=profiler,
    )


def tokenize_parsed_scores(
    parsed_scores: Iterable[ParsedScoreArtifact],
    *,
    dataset_root: Path,
    paths: ProcessedDatasetPaths,
    snapshot: TokenizerSnapshot,
    segmentation_config: SegmentationConfig,
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
    difficulty_labels: dict[str, int | None] | None,
    data_processing_config: DataProcessingConfig,
    overwrite: bool,
    show_progress: bool,
    profiler: ProcessingProfilerProtocol = NULL_PROCESSING_PROFILER,
) -> TokenizeDatasetResult:
    parsed_scores = tuple(parsed_scores)
    encoded_jsonl_path = paths.encoded_jsonl_path(snapshot.tokenizer_hash)
    encoded_manifest_path = paths.encoded_manifest_path(snapshot.tokenizer_hash)
    tokenizer_snapshot_path = paths.tokenizer_snapshot_path(snapshot.tokenizer_hash)
    state_path = paths.tokenization_state_path(snapshot.tokenizer_hash)
    state_key = _tokenization_state_key(
        snapshot=snapshot,
        segmentation_config=segmentation_config,
        data_processing_config=data_processing_config,
        difficulty_labels=difficulty_labels,
    )

    if overwrite:
        _clear_tokenization_outputs(
            encoded_jsonl_path=encoded_jsonl_path,
            encoded_manifest_path=encoded_manifest_path,
            tokenizer_snapshot_path=tokenizer_snapshot_path,
            state_path=state_path,
        )

    resume_state = _load_tokenization_resume_state(state_path, state_key=state_key)
    if not resume_state.state_key_matches:
        _LOGGER.warning("Ignoring stale tokenization resume state for %s", state_path.parent)
        _clear_tokenization_outputs(
            encoded_jsonl_path=encoded_jsonl_path,
            encoded_manifest_path=encoded_manifest_path,
            tokenizer_snapshot_path=tokenizer_snapshot_path,
            state_path=state_path,
        )
        resume_state = _load_tokenization_resume_state(state_path, state_key=state_key)
    if _resume_state_outputs_missing(
        resume_state=resume_state,
        encoded_jsonl_path=encoded_jsonl_path,
        encoded_manifest_path=encoded_manifest_path,
    ):
        _LOGGER.warning("Ignoring tokenization resume state with missing outputs for %s", state_path.parent)
        _clear_tokenization_outputs(
            encoded_jsonl_path=encoded_jsonl_path,
            encoded_manifest_path=encoded_manifest_path,
            tokenizer_snapshot_path=tokenizer_snapshot_path,
            state_path=state_path,
        )
        resume_state = _load_tokenization_resume_state(state_path, state_key=state_key)

    if _complete_encoded_outputs_exist(
        parsed_scores=parsed_scores,
        encoded_jsonl_path=encoded_jsonl_path,
        encoded_manifest_path=encoded_manifest_path,
        tokenizer_snapshot_path=tokenizer_snapshot_path,
        resume_state=resume_state,
    ):
        encoded_rows = read_encoded_manifest(encoded_manifest_path)
        encoded_count = sum(1 for row in encoded_rows if row[EncodedManifestField.ENCODED_LINE] != "")
        _LOGGER.info("Reusing complete encoded artifacts: %s", encoded_manifest_path)
        return TokenizeDatasetResult(
            parsed_manifest_path=paths.parsed_manifest_path,
            encoded_manifest_path=encoded_manifest_path,
            tokenizer_snapshot_path=tokenizer_snapshot_path,
            encoded_count=encoded_count,
            segment_count=len(encoded_rows),
            scale_match_support_score_margin=data_processing_config.scale_match_support_score_margin,
            scale_match_selection_score_margin=data_processing_config.scale_match_selection_score_margin,
            scale_match_maximum_unexplained_weight_fraction=(
                data_processing_config.scale_match_maximum_unexplained_weight_fraction
            ),
            scale_match_maximum_explanation_pitch_class_count=(
                data_processing_config.scale_match_maximum_explanation_pitch_class_count
            ),
        )

    with profiler.measure("prepare_encoded_outputs"):
        _prepare_tokenization_resume_outputs(
            encoded_jsonl_path=encoded_jsonl_path,
            encoded_manifest_path=encoded_manifest_path,
            state_path=state_path,
            state_key=state_key,
            resume_state=resume_state,
        )

    _LOGGER.info("Encoding %s parsed score(s)", len(parsed_scores))
    _log_difficulty_label_stats(parsed_scores, dataset_root=dataset_root, difficulty_labels=difficulty_labels)
    with profiler.measure("write_tokenizer_snapshot"):
        write_json_model(snapshot, tokenizer_snapshot_path, overwrite=True)

    completed_source_ids = set(resume_state.completed_source_ids)
    encoded_line_count = resume_state.encoded_line_count
    manifest_row_count = resume_state.manifest_row_count
    encoded_count = resume_state.encoded_count
    for artifact in _progress(parsed_scores, description="Encoding scores", unit="score", enabled=show_progress):
        if artifact.source_id_value in completed_source_ids:
            continue

        source_encoded_count, source_manifest_count, encoded_line_count = _tokenize_source(
            artifact,
            dataset_root=dataset_root,
            paths=paths,
            encoded_jsonl_path=encoded_jsonl_path,
            encoded_manifest_path=encoded_manifest_path,
            segmentation_config=segmentation_config,
            duration_vocabulary=duration_vocabulary,
            token_vocabulary=token_vocabulary,
            difficulty_labels=difficulty_labels,
            data_processing_config=data_processing_config,
            encoded_line_count=encoded_line_count,
            profiler=profiler,
        )
        encoded_count += source_encoded_count
        manifest_row_count += source_manifest_count
        _append_tokenization_state_event(
            state_path,
            {
                "type": _TOKENIZATION_SOURCE_COMPLETED,
                "state_key": state_key,
                "source_id": artifact.source_id_value,
                "encoded_line_count": encoded_line_count,
                "manifest_row_count": manifest_row_count,
                "encoded_count": encoded_count,
            },
        )
        completed_source_ids.add(artifact.source_id_value)

    _LOGGER.info("Wrote encoded manifest: %s", encoded_manifest_path)
    return TokenizeDatasetResult(
        parsed_manifest_path=paths.parsed_manifest_path,
        encoded_manifest_path=encoded_manifest_path,
        tokenizer_snapshot_path=tokenizer_snapshot_path,
        encoded_count=encoded_count,
        segment_count=manifest_row_count,
        scale_match_support_score_margin=data_processing_config.scale_match_support_score_margin,
        scale_match_selection_score_margin=data_processing_config.scale_match_selection_score_margin,
        scale_match_maximum_unexplained_weight_fraction=(
            data_processing_config.scale_match_maximum_unexplained_weight_fraction
        ),
        scale_match_maximum_explanation_pitch_class_count=(
            data_processing_config.scale_match_maximum_explanation_pitch_class_count
        ),
    )


def _tokenize_source(
    artifact: ParsedScoreArtifact,
    *,
    dataset_root: Path,
    paths: ProcessedDatasetPaths,
    encoded_jsonl_path: Path,
    encoded_manifest_path: Path,
    segmentation_config: SegmentationConfig,
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
    difficulty_labels: dict[str, int | None] | None,
    data_processing_config: DataProcessingConfig,
    encoded_line_count: int,
    profiler: ProcessingProfilerProtocol,
) -> tuple[int, int, int]:
    source_metadata_path = Path(artifact.source_path.resolve().relative_to(dataset_root.resolve()).as_posix())
    segments = segment_parsed_score(
        artifact.score,
        source_metadata_path,
        duration_vocabulary,
        segmentation_config=segmentation_config,
        difficulty_labels=difficulty_labels,
        scale_match_support_score_margin=data_processing_config.scale_match_support_score_margin,
        scale_match_selection_score_margin=data_processing_config.scale_match_selection_score_margin,
        scale_match_maximum_unexplained_weight_fraction=(
            data_processing_config.scale_match_maximum_unexplained_weight_fraction
        ),
        scale_match_maximum_explanation_pitch_class_count=(
            data_processing_config.scale_match_maximum_explanation_pitch_class_count
        ),
        profiler=profiler,
    )

    source_rows: list[dict[str, object]] = []
    source_encoded_count = 0
    for segment in segments:
        with profiler.measure("apply_processing_filters", source_file=source_metadata_path):
            segment = _apply_processing_filters(
                segment,
                duration_vocabulary=duration_vocabulary,
                data_processing_config=data_processing_config,
            )
        encoded_sample = None
        if segment.metadata.eligible_for_training:
            with profiler.measure("encode_segment", source_file=source_metadata_path):
                encoded_sample = _encode_segment(segment, token_vocabulary=token_vocabulary)
        encoded_line = None
        if encoded_sample is not None:
            with profiler.measure("append_encoded_jsonl", source_file=source_metadata_path):
                encoded_line = _append_jsonl_model(encoded_sample, encoded_jsonl_path, line_index=encoded_line_count)
            encoded_line_count += 1
            source_encoded_count += 1
        with profiler.measure("encoded_manifest_row", source_file=source_metadata_path):
            source_rows.append(
                encoded_row(
                    source_id_value=artifact.source_id_value,
                    source_path=artifact.source_path,
                    dataset_root=dataset_root,
                    parsed_path=artifact.parsed_path,
                    processed_root=paths.root,
                    segment=segment,
                    duration_vocabulary=duration_vocabulary,
                    encoded_sample=encoded_sample,
                    encoded_shard=encoded_jsonl_path,
                    encoded_line=encoded_line,
                    segmentation_mode=segmentation_config.mode,
                )
            )

    with profiler.measure("append_encoded_manifest", source_file=source_metadata_path):
        _append_encoded_manifest_rows(source_rows, encoded_manifest_path)
    return source_encoded_count, len(source_rows), encoded_line_count


def _apply_processing_filters(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
    data_processing_config: DataProcessingConfig,
) -> Segment:
    if not data_processing_config.remove_segments_with_silent_bars:
        return segment

    diagnostics = diagnose_segment(segment, duration_vocabulary=duration_vocabulary)
    if diagnostics.silent_bar_count == 0:
        return segment

    metadata = segment.metadata.model_copy(
        update={
            "eligible_for_training": False,
            "ineligibility_reasons": segment.metadata.ineligibility_reasons
            | frozenset({SegmentIneligibilityReason.SILENT_BAR}),
        }
    )
    return segment.model_copy(update={"metadata": metadata})


def _complete_encoded_outputs_exist(
    *,
    parsed_scores: tuple[ParsedScoreArtifact, ...],
    encoded_jsonl_path: Path,
    encoded_manifest_path: Path,
    tokenizer_snapshot_path: Path,
    resume_state: _TokenizationResumeState,
) -> bool:
    if not (encoded_jsonl_path.exists() and encoded_manifest_path.exists() and tokenizer_snapshot_path.exists()):
        return False

    source_ids = {artifact.source_id_value for artifact in parsed_scores}
    return source_ids <= resume_state.completed_source_ids or not resume_state.completed_source_ids


def _resume_state_outputs_missing(
    *,
    resume_state: _TokenizationResumeState,
    encoded_jsonl_path: Path,
    encoded_manifest_path: Path,
) -> bool:
    if resume_state.encoded_line_count == 0 and resume_state.manifest_row_count == 0:
        return False

    return not (encoded_jsonl_path.exists() and encoded_manifest_path.exists())


def _prepare_tokenization_resume_outputs(
    *,
    encoded_jsonl_path: Path,
    encoded_manifest_path: Path,
    state_path: Path,
    state_key: str,
    resume_state: _TokenizationResumeState,
) -> None:
    if not state_path.exists():
        _clear_tokenization_outputs(
            encoded_jsonl_path=encoded_jsonl_path,
            encoded_manifest_path=encoded_manifest_path,
            tokenizer_snapshot_path=None,
            state_path=state_path,
        )
        _append_tokenization_state_event(
            state_path,
            {"type": _TOKENIZATION_STATE_HEADER, "version": _TOKENIZATION_STATE_VERSION, "state_key": state_key},
        )
        return

    _truncate_text_lines(encoded_jsonl_path, resume_state.encoded_line_count)
    _truncate_manifest_rows(encoded_manifest_path, resume_state.manifest_row_count)


def _clear_tokenization_outputs(
    *,
    encoded_jsonl_path: Path,
    encoded_manifest_path: Path,
    tokenizer_snapshot_path: Path | None,
    state_path: Path,
) -> None:
    for path in (encoded_jsonl_path, encoded_manifest_path, tokenizer_snapshot_path, state_path):
        if path is not None:
            path.unlink(missing_ok=True)


def _load_tokenization_resume_state(path: Path, *, state_key: str) -> _TokenizationResumeState:
    if not path.exists():
        return _TokenizationResumeState(
            completed_source_ids=frozenset(),
            encoded_line_count=0,
            manifest_row_count=0,
            encoded_count=0,
            state_key_matches=True,
        )

    completed_source_ids: set[str] = set()
    encoded_line_count = 0
    manifest_row_count = 0
    encoded_count = 0
    state_key_matches = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "":
            continue

        event = json.loads(line)
        if event.get("state_key") != state_key:
            return _TokenizationResumeState(
                completed_source_ids=frozenset(),
                encoded_line_count=0,
                manifest_row_count=0,
                encoded_count=0,
                state_key_matches=False,
            )

        if event["type"] == _TOKENIZATION_STATE_HEADER:
            state_key_matches = event.get("version") == _TOKENIZATION_STATE_VERSION
            continue

        if event["type"] == _TOKENIZATION_SOURCE_COMPLETED:
            completed_source_ids.add(event["source_id"])
            encoded_line_count = int(event["encoded_line_count"])
            manifest_row_count = int(event["manifest_row_count"])
            encoded_count = int(event["encoded_count"])

    return _TokenizationResumeState(
        completed_source_ids=frozenset(completed_source_ids),
        encoded_line_count=encoded_line_count,
        manifest_row_count=manifest_row_count,
        encoded_count=encoded_count,
        state_key_matches=state_key_matches,
    )


def _append_tokenization_state_event(path: Path, event: dict[str, int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, sort_keys=True))
        file.write("\n")


def _tokenization_state_key(
    *,
    snapshot: TokenizerSnapshot,
    segmentation_config: SegmentationConfig,
    data_processing_config: DataProcessingConfig,
    difficulty_labels: dict[str, int | None] | None,
) -> str:
    payload = {
        "tokenizer_hash": snapshot.tokenizer_hash,
        "segmentation": segmentation_config.model_dump(mode="json"),
        "data_processing": data_processing_config.model_dump(mode="json"),
        "difficulty_labels": difficulty_labels or {},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _append_jsonl_model(model: BaseModel, path: Path, *, line_index: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(model.model_dump_json())
        file.write("\n")

    return line_index


def _append_encoded_manifest_rows(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[field.value for field in ENCODED_MANIFEST_FIELDS])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _truncate_text_lines(path: Path, line_count: int) -> None:
    if not path.exists():
        return

    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("".join(f"{line}\n" for line in lines[:line_count]), encoding="utf-8")


def _truncate_manifest_rows(path: Path, row_count: int) -> None:
    if not path.exists():
        return

    rows = read_encoded_manifest(path)[:row_count]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[field.value for field in ENCODED_MANIFEST_FIELDS])
        writer.writeheader()
        writer.writerows(rows)


def _log_difficulty_label_stats(
    parsed_scores: tuple[ParsedScoreArtifact, ...],
    *,
    dataset_root: Path,
    difficulty_labels: dict[str, int | None] | None,
) -> None:
    if difficulty_labels is None:
        return

    stats = _difficulty_label_stats(
        parsed_scores,
        dataset_root=dataset_root,
        difficulty_labels=difficulty_labels,
    )
    message = (
        f"Difficulty labels: labeled={stats.labeled} "
        f"explicit_unlabeled={stats.explicit_unlabeled} "
        f"unspecified={stats.unspecified}"
    )
    if stats.unspecified > 0:
        _LOGGER.warning(message)
    else:
        _LOGGER.info(message)


def _difficulty_label_stats(
    parsed_scores: tuple[ParsedScoreArtifact, ...],
    *,
    dataset_root: Path,
    difficulty_labels: dict[str, int | None],
) -> _DifficultyLabelStats:
    labeled = 0
    explicit_unlabeled = 0
    unspecified = 0
    for artifact in parsed_scores:
        relative_source_path = Path(artifact.source_path.resolve().relative_to(dataset_root.resolve()).as_posix())
        matching_key = _first_difficulty_label_key(relative_source_path, difficulty_labels)
        if matching_key is None:
            unspecified += 1
            continue

        if difficulty_labels[matching_key] is None:
            explicit_unlabeled += 1
        else:
            labeled += 1

    return _DifficultyLabelStats(
        labeled=labeled,
        explicit_unlabeled=explicit_unlabeled,
        unspecified=unspecified,
    )


def _first_difficulty_label_key(path: Path, labels: dict[str, int | None]) -> str | None:
    for key in (path.as_posix(), path.name, path.stem):
        if key in labels:
            return key

    return None


def _progress(
    values: Iterable[_T],
    *,
    description: str,
    unit: str,
    enabled: bool,
    total: int | None = None,
) -> Iterable[_T]:
    if not enabled:
        return values

    return cast(
        Iterable[_T],
        tqdm(
            values,
            total=total,
            desc=description,
            unit=unit,
        ),
    )
