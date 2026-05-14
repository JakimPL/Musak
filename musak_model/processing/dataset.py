from __future__ import annotations

import logging
from collections.abc import Iterable
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar, cast
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, ParseError
from zipfile import BadZipFile, ZipFile

from music21.exceptions21 import Music21Exception
from tqdm.auto import tqdm

from musak_model.common.files import collect_musicxml_files
from musak_model.data.cleaning import clean_parsed_score
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
_T = TypeVar("_T")
_LOGGER = logging.getLogger(__name__)

_PROCESSING_ERRORS: tuple[type[Exception], ...] = (
    Music21Exception,
    OSError,
    OverflowError,
    ParseError,
    BadZipFile,
    TypeError,
    ValueError,
)

_TITLE_EXTRACTION_ERRORS: tuple[type[Exception], ...] = (
    KeyError,
    OSError,
    ParseError,
    BadZipFile,
)

_MXL_CONTAINER_PATH = "META-INF/container.xml"
_MXL_SUFFIX = ".mxl"
_MUSICXML_TITLE_FIELDS = ("movement-title", "work-title")


@dataclass(frozen=True)
class ProcessDatasetResult:
    parsed_manifest_path: Path
    encoded_manifest_path: Path | None
    tokenizer_snapshot_path: Path | None
    parsed_count: int
    encoded_count: int
    error_count: int


@dataclass(frozen=True)
class _ParsedScoreTask:
    index: int
    source_path: Path
    dataset_root: Path
    paths: ProcessedDatasetPaths
    overwrite: bool


@dataclass(frozen=True)
class _ParsedScoreResult:
    index: int
    source_id_value: str
    source_path: Path
    parsed_path: Path
    row: dict[str, object]
    score: ParsedScore | None


def process_dataset(
    dataset_root: Path,
    *,
    processed_root: Path,
    segmentation: SegmentationConfig,
    tokenization_config: TokenizationConfig,
    stage: ProcessingStage = "all",
    difficulty_labels: dict[str, int] | None = None,
    overwrite: bool = False,
    workers: int = 1,
    show_progress: bool = False,
) -> ProcessDatasetResult:
    if workers < 1:
        raise ValueError("workers must be >= 1")

    paths = ProcessedDatasetPaths.from_dataset_root(processed_root=processed_root, dataset_root=dataset_root)
    _LOGGER.info("Processing dataset from %s into %s", dataset_root, paths.root)
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
        workers=workers,
        show_progress=show_progress,
    )
    write_parsed_manifest(parsed_rows, paths.parsed_manifest_path)
    _LOGGER.info("Wrote parsed manifest: %s", paths.parsed_manifest_path)

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
            show_progress=show_progress,
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
    workers: int,
    show_progress: bool,
) -> tuple[list[dict[str, object]], list[tuple[str, Path, Path, ParsedScore]]]:
    rows: list[dict[str, object]] = []
    parsed_scores: list[tuple[str, Path, Path, ParsedScore]] = []
    tasks = [
        _ParsedScoreTask(
            index=index,
            source_path=source_path,
            dataset_root=dataset_root,
            paths=paths,
            overwrite=overwrite,
        )
        for index, source_path in enumerate(collect_musicxml_files(dataset_root))
    ]
    _LOGGER.info("Parsing %s source file(s) with %s worker(s)", len(tasks), workers)
    results = _collect_parsed_score_results(tasks, workers=workers, show_progress=show_progress)
    for result in results:
        rows.append(result.row)
        if result.score is not None:
            parsed_scores.append((result.source_id_value, result.source_path, result.parsed_path, result.score))

    _LOGGER.info("Parsed %s/%s source file(s)", len(parsed_scores), len(tasks))
    return rows, parsed_scores


def _collect_parsed_score_results(
    tasks: list[_ParsedScoreTask],
    *,
    workers: int,
    show_progress: bool,
) -> list[_ParsedScoreResult]:
    ordered_results: list[_ParsedScoreResult | None] = [None] * len(tasks)
    if workers == 1:
        for task in _progress(tasks, description="Parsing scores", unit="score", enabled=show_progress):
            result = _process_parsed_score_task(task)
            ordered_results[result.index] = result
        return _filled_results(ordered_results)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures: dict[Future[_ParsedScoreResult], int] = {
            executor.submit(_process_parsed_score_task, task): task.index for task in tasks
        }
        completed_futures = as_completed(futures)
        for future in _progress(
            completed_futures,
            total=len(futures),
            description="Parsing scores",
            unit="score",
            enabled=show_progress,
        ):
            result = future.result()
            ordered_results[result.index] = result

    return _filled_results(ordered_results)


def _filled_results(results: list[_ParsedScoreResult | None]) -> list[_ParsedScoreResult]:
    return [result for result in results if result is not None]


def _process_parsed_score_task(task: _ParsedScoreTask) -> _ParsedScoreResult:
    source_id_value = source_id(task.source_path, dataset_root=task.dataset_root)
    parsed_path = task.paths.parsed_score_path(source_id_value)
    title = _score_title(task.source_path)
    try:
        if parsed_path.exists() and not task.overwrite:
            score = load_parsed_score_json(parsed_path)
        else:
            score = clean_parsed_score(parse_score(task.source_path))
            write_json_model(score, parsed_path, overwrite=task.overwrite)
    except _PROCESSING_ERRORS as exception:
        return _ParsedScoreResult(
            index=task.index,
            source_id_value=source_id_value,
            source_path=task.source_path,
            parsed_path=parsed_path,
            row=parsed_error_row(
                source_id_value=source_id_value,
                source_path=task.source_path,
                dataset_root=task.dataset_root,
                title=title,
                exception=exception,
            ),
            score=None,
        )

    return _ParsedScoreResult(
        index=task.index,
        source_id_value=source_id_value,
        source_path=task.source_path,
        parsed_path=parsed_path,
        row=parsed_success_row(
            source_id_value=source_id_value,
            source_path=task.source_path,
            dataset_root=task.dataset_root,
            title=title,
            parsed_path=parsed_path,
            processed_root=task.paths.root,
            score=score,
        ),
        score=score,
    )


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
    show_progress: bool,
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
        _LOGGER.info("Reusing complete encoded artifacts: %s", encoded_manifest_path)
        return encoded_count, encoded_manifest_path, tokenizer_snapshot_path

    _prepare_encoded_outputs(
        encoded_jsonl_path=encoded_jsonl_path,
        encoded_manifest_path=encoded_manifest_path,
        tokenizer_snapshot_path=tokenizer_snapshot_path,
        overwrite=overwrite,
    )

    _LOGGER.info("Encoding %s parsed score(s)", len(parsed_scores))
    write_json_model(snapshot, tokenizer_snapshot_path, overwrite=True)
    rows: list[dict[str, object]] = []
    encoded_count = 0
    for source_id_value, source_path, parsed_path, score in _progress(
        parsed_scores,
        description="Encoding scores",
        unit="score",
        enabled=show_progress,
    ):
        source_metadata_path = Path(source_path.resolve().relative_to(dataset_root.resolve()).as_posix())
        segments = segment_parsed_score(
            score,
            source_metadata_path,
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
    _LOGGER.info("Wrote encoded manifest: %s", encoded_manifest_path)
    return encoded_count, encoded_manifest_path, tokenizer_snapshot_path


def _prepare_encoded_outputs(
    *,
    encoded_jsonl_path: Path,
    encoded_manifest_path: Path,
    tokenizer_snapshot_path: Path,
    overwrite: bool,
) -> None:
    existing_paths = (encoded_jsonl_path, encoded_manifest_path, tokenizer_snapshot_path)
    if overwrite:
        for path in existing_paths:
            path.unlink(missing_ok=True)
        return

    incomplete_paths = [path for path in existing_paths if path.exists()]
    if not incomplete_paths:
        return

    _LOGGER.warning(
        "Found incomplete encoded artifacts for %s; rebuilding encoded outputs from parsed scores",
        encoded_jsonl_path.parent,
    )
    for path in incomplete_paths:
        path.unlink()


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

    return cast(Iterable[_T], tqdm(values, total=total, desc=description, unit=unit))


def _score_title(path: Path) -> str:
    try:
        root = _musicxml_root(path)
    except _TITLE_EXTRACTION_ERRORS:
        return ""

    for field in _MUSICXML_TITLE_FIELDS:
        title = _first_musicxml_text(root, field)
        if title != "":
            return title

    return ""


def _musicxml_root(path: Path) -> Element:
    if path.suffix.lower() == _MXL_SUFFIX:
        return _compressed_musicxml_root(path)

    return ElementTree.parse(path).getroot()


def _compressed_musicxml_root(path: Path) -> Element:
    with ZipFile(path) as archive:
        rootfile_path = _mxl_rootfile_path(archive)
        with archive.open(rootfile_path) as file:
            return ElementTree.parse(file).getroot()


def _mxl_rootfile_path(archive: ZipFile) -> str:
    with archive.open(_MXL_CONTAINER_PATH) as file:
        container = ElementTree.parse(file).getroot()

    for element in container.iter():
        if _xml_local_name(element.tag) != "rootfile":
            continue

        full_path = element.attrib.get("full-path")
        if full_path is not None and full_path != "":
            return full_path

    raise KeyError("missing MusicXML rootfile in MXL container")


def _first_musicxml_text(root: Element, field_name: str) -> str:
    for element in root.iter():
        if _xml_local_name(element.tag) != field_name or element.text is None:
            continue

        title = element.text.strip()
        if title != "":
            return title

    return ""


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]
