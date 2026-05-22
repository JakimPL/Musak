import logging
import multiprocessing
from collections.abc import Iterable
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from enum import StrEnum
from multiprocessing.context import BaseContext
from pathlib import Path
from typing import TypeVar, cast
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, ParseError
from zipfile import BadZipFile, ZipFile

from music21.exceptions21 import Music21Exception
from tqdm.auto import tqdm

from musak_model.data.cleaning import clean_parsed_score
from musak_model.data.parser import parse_score
from musak_model.data.schema import ParsedScore
from musak_model.processing.diagnostics import ParseDiagnosticsCapture
from musak_model.processing.ids import source_id
from musak_model.processing.io import load_parsed_score_json, write_json_model
from musak_model.processing.manifest import (
    ParsedManifestField,
    ParsedManifestStatus,
    parsed_error_row,
    parsed_success_row,
    read_parsed_manifest,
    write_parsed_manifest,
)
from musak_model.processing.paths import ProcessedDatasetPaths
from musak_model.processing.profiler import NULL_PROCESSING_PROFILER, ProcessingProfilerProtocol
from musak_shared.files import collect_musicxml_files

_T = TypeVar("_T")
_LOGGER = logging.getLogger(__name__)

_MXL_CONTAINER_PATH = "META-INF/container.xml"
_MXL_SUFFIX = ".mxl"
_MUSICXML_TITLE_FIELDS = ("movement-title", "work-title")


@dataclass(frozen=True)
class ParsedScoreArtifact:
    source_id_value: str
    source_path: Path
    parsed_path: Path
    score: ParsedScore


@dataclass(frozen=True)
class ParseDatasetResult:
    parsed_manifest_path: Path
    parsed_count: int
    error_count: int
    parsed_scores: tuple[ParsedScoreArtifact, ...]


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


class _ParsedManifestReuseIssue(StrEnum):
    UNSUPPORTED_STATUS = "unsupported_status"
    MISSING_PARSED_PATH = "missing_parsed_path"
    UNREADABLE_PARSED_SCORE = "unreadable_parsed_score"


def parse_dataset(
    dataset_root: Path,
    *,
    processed_root: Path,
    overwrite: bool,
    workers: int,
    show_progress: bool,
    profiler: ProcessingProfilerProtocol = NULL_PROCESSING_PROFILER,
) -> ParseDatasetResult:
    if workers < 1:
        raise ValueError("workers must be >= 1")

    paths = ProcessedDatasetPaths.from_dataset_root(processed_root=processed_root, dataset_root=dataset_root)
    _LOGGER.info("Parsing dataset from %s into %s", dataset_root, paths.root)
    with profiler.measure("process_parsed_scores"):
        rows, parsed_scores = _process_parsed_scores(
            dataset_root,
            paths=paths,
            overwrite=overwrite,
            workers=workers,
            show_progress=show_progress,
        )
    _LOGGER.info("Writing parsed manifest to %s", paths.parsed_manifest_path)
    with profiler.measure("write_parsed_manifest"):
        write_parsed_manifest(rows, paths.parsed_manifest_path)
    _LOGGER.info("Wrote parsed manifest: %s", paths.parsed_manifest_path)

    return ParseDatasetResult(
        parsed_manifest_path=paths.parsed_manifest_path,
        parsed_count=sum(1 for row in rows if row[ParsedManifestField.STATUS] == ParsedManifestStatus.SUCCESS.value),
        error_count=sum(1 for row in rows if row[ParsedManifestField.STATUS] == ParsedManifestStatus.ERROR.value),
        parsed_scores=tuple(parsed_scores),
    )


def load_parsed_score_artifacts(
    dataset_root: Path,
    *,
    processed_root: Path,
) -> tuple[ParsedScoreArtifact, ...]:
    paths = ProcessedDatasetPaths.from_dataset_root(processed_root=processed_root, dataset_root=dataset_root)
    if not paths.parsed_manifest_path.exists():
        raise FileNotFoundError(f"parsed manifest does not exist: {paths.parsed_manifest_path}")

    artifacts: list[ParsedScoreArtifact] = []
    for row in read_parsed_manifest(paths.parsed_manifest_path):
        if row[ParsedManifestField.STATUS] != ParsedManifestStatus.SUCCESS.value:
            continue

        parsed_path_text = row[ParsedManifestField.PARSED_PATH]
        if parsed_path_text == "":
            raise ValueError(f"parsed manifest row is missing parsed_path: {row[ParsedManifestField.SOURCE_ID]}")

        parsed_path = paths.root / parsed_path_text
        if not parsed_path.exists():
            raise FileNotFoundError(f"parsed score artifact does not exist: {parsed_path}")

        source_path = dataset_root / row[ParsedManifestField.SOURCE_PATH]
        artifacts.append(
            ParsedScoreArtifact(
                source_id_value=row[ParsedManifestField.SOURCE_ID],
                source_path=source_path,
                parsed_path=parsed_path,
                score=load_parsed_score_json(parsed_path),
            )
        )

    return tuple(artifacts)


def _process_parsed_scores(
    dataset_root: Path,
    *,
    paths: ProcessedDatasetPaths,
    overwrite: bool,
    workers: int,
    show_progress: bool,
) -> tuple[list[dict[str, object]], list[ParsedScoreArtifact]]:
    source_paths = collect_musicxml_files(dataset_root)
    reusable_rows = _reusable_parsed_manifest_rows(
        paths=paths,
        overwrite=overwrite,
    )
    ordered_results: list[_ParsedScoreResult | None] = [None] * len(source_paths)
    tasks: list[_ParsedScoreTask] = []
    reused_success_count = 0
    reused_error_count = 0
    reuse_issues: dict[_ParsedManifestReuseIssue, int] = {}

    for index, source_path in enumerate(source_paths):
        source_id_value = source_id(source_path, dataset_root=dataset_root)
        reused_result, reuse_issue = _reused_parsed_result(
            index=index,
            source_id_value=source_id_value,
            source_path=source_path,
            paths=paths,
            row=reusable_rows.get(source_id_value),
        )
        if reuse_issue is not None:
            reuse_issues[reuse_issue] = reuse_issues.get(reuse_issue, 0) + 1

        if reused_result is not None:
            ordered_results[index] = reused_result
            if reused_result.score is None:
                reused_error_count += 1
            else:
                reused_success_count += 1
            continue

        tasks.append(
            _ParsedScoreTask(
                index=index,
                source_path=source_path,
                dataset_root=dataset_root,
                paths=paths,
                overwrite=overwrite,
            )
        )

    reused_count = len(source_paths) - len(tasks)
    if reused_count:
        _LOGGER.info(
            "Reusing %s parsed manifest row(s): %s success(es), %s error(s)",
            reused_count,
            reused_success_count,
            reused_error_count,
        )
    for issue, count in reuse_issues.items():
        _LOGGER.warning("Could not reuse %s parsed manifest row(s): %s", count, issue.value)
    _LOGGER.info("Parsing %s/%s source file(s) with %s worker(s)", len(tasks), len(source_paths), workers)

    completed = False
    try:
        _run_parsed_score_tasks(
            tasks,
            workers=workers,
            show_progress=show_progress,
            ordered_results=ordered_results,
        )
        completed = True
    finally:
        partial_rows = _parsed_rows_from_results(ordered_results)
        if partial_rows and not completed:
            _LOGGER.info("Writing partial parsed manifest to %s", paths.parsed_manifest_path)
            write_parsed_manifest(partial_rows, paths.parsed_manifest_path)
            _LOGGER.info("Wrote partial parsed manifest with %s row(s)", len(partial_rows))

    results = _filled_results(ordered_results)
    rows = [result.row for result in results]
    parsed_scores = [
        ParsedScoreArtifact(
            source_id_value=result.source_id_value,
            source_path=result.source_path,
            parsed_path=result.parsed_path,
            score=result.score,
        )
        for result in results
        if result.score is not None
    ]

    _LOGGER.info("Parsed %s/%s source file(s)", len(parsed_scores), len(source_paths))
    return rows, parsed_scores


def _run_parsed_score_tasks(
    tasks: list[_ParsedScoreTask],
    *,
    workers: int,
    show_progress: bool,
    ordered_results: list[_ParsedScoreResult | None],
) -> None:
    if not tasks:
        return

    if workers == 1:
        for task in _progress(tasks, description="Parsing scores", unit="score", enabled=show_progress):
            result = _process_parsed_score_task(task)
            ordered_results[result.index] = result
        return

    with ProcessPoolExecutor(max_workers=workers, mp_context=_process_pool_context()) as executor:
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


def _process_pool_context() -> BaseContext:
    available_methods = multiprocessing.get_all_start_methods()
    if "forkserver" in available_methods:
        return multiprocessing.get_context("forkserver")

    if "spawn" in available_methods:
        return multiprocessing.get_context("spawn")

    return multiprocessing.get_context()


def _filled_results(results: list[_ParsedScoreResult | None]) -> list[_ParsedScoreResult]:
    return [result for result in results if result is not None]


def _parsed_rows_from_results(results: list[_ParsedScoreResult | None]) -> list[dict[str, object]]:
    return [result.row for result in results if result is not None]


def _reusable_parsed_manifest_rows(
    *,
    paths: ProcessedDatasetPaths,
    overwrite: bool,
) -> dict[str, dict[str, str]]:
    if overwrite or not paths.parsed_manifest_path.exists():
        return {}

    rows = read_parsed_manifest(paths.parsed_manifest_path)
    _LOGGER.info("Loaded parsed manifest for resume: %s (%s row(s))", paths.parsed_manifest_path, len(rows))
    return {row[ParsedManifestField.SOURCE_ID]: row for row in rows if row.get(ParsedManifestField.SOURCE_ID, "")}


def _reused_parsed_result(
    *,
    index: int,
    source_id_value: str,
    source_path: Path,
    paths: ProcessedDatasetPaths,
    row: dict[str, str] | None,
) -> tuple[_ParsedScoreResult | None, _ParsedManifestReuseIssue | None]:
    if row is None:
        return None, None

    parsed_path = paths.parsed_score_path(source_id_value)
    status = row[ParsedManifestField.STATUS]
    if status == ParsedManifestStatus.ERROR.value:
        return (
            _ParsedScoreResult(
                index=index,
                source_id_value=source_id_value,
                source_path=source_path,
                parsed_path=parsed_path,
                row=dict(row),
                score=None,
            ),
            None,
        )

    if status != ParsedManifestStatus.SUCCESS.value:
        return None, _ParsedManifestReuseIssue.UNSUPPORTED_STATUS

    parsed_path_text = row[ParsedManifestField.PARSED_PATH]
    if parsed_path_text == "":
        return None, _ParsedManifestReuseIssue.MISSING_PARSED_PATH

    parsed_path = paths.root / parsed_path_text
    try:
        score = load_parsed_score_json(parsed_path)
    except (OSError, ValueError):
        return None, _ParsedManifestReuseIssue.UNREADABLE_PARSED_SCORE

    return (
        _ParsedScoreResult(
            index=index,
            source_id_value=source_id_value,
            source_path=source_path,
            parsed_path=parsed_path,
            row=dict(row),
            score=score,
        ),
        None,
    )


def _process_parsed_score_task(task: _ParsedScoreTask) -> _ParsedScoreResult:
    source_id_value = source_id(task.source_path, dataset_root=task.dataset_root)
    parsed_path = task.paths.parsed_score_path(source_id_value)
    title = _score_title(task.source_path)
    diagnostics = ""
    captured_diagnostics: ParseDiagnosticsCapture | None = None
    try:
        if parsed_path.exists() and not task.overwrite:
            score = load_parsed_score_json(parsed_path)
        else:
            with ParseDiagnosticsCapture() as captured_diagnostics:
                score = clean_parsed_score(parse_score(task.source_path))
            diagnostics = captured_diagnostics.text()
            write_json_model(score, parsed_path, overwrite=task.overwrite)
    except (Music21Exception, OSError, OverflowError, ParseError, BadZipFile, TypeError, ValueError) as exception:
        if captured_diagnostics is not None:
            diagnostics = captured_diagnostics.text()
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
                parse_diagnostics=diagnostics,
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
            parse_diagnostics=diagnostics,
        ),
        score=score,
    )


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
    except (KeyError, OSError, ParseError, BadZipFile):
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
