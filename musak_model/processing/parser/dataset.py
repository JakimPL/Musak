import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from musak_model.processing.ids import source_id
from musak_model.processing.manifest import ParsedManifestField, ParsedManifestStatus, write_parsed_manifest
from musak_model.processing.parser.manifest import (
    ParsedManifestReuseIssue,
    reusable_parsed_manifest_rows,
    reused_parsed_result,
)
from musak_model.processing.parser.schema import ParseDatasetResult, ParsedScoreResult, ParsedScoreTask
from musak_model.processing.parser.worker import run_parsed_score_tasks
from musak_model.processing.paths import ProcessedDatasetPaths
from musak_shared.files import collect_musicxml_files
from musak_shared.profiling import NULL_PROFILER, ProfilerProtocol

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ParsePlan:
    ordered_results: list[ParsedScoreResult | None]
    tasks: list[ParsedScoreTask]


@dataclass(frozen=True)
class _ReuseStats:
    success_count: int
    error_count: int
    issues: dict[ParsedManifestReuseIssue, int]


def parse_dataset(
    dataset_root: Path,
    *,
    processed_root: Path,
    overwrite: bool,
    workers: int,
    show_progress: bool,
    profiler: ProfilerProtocol = NULL_PROFILER,
) -> ParseDatasetResult:
    if workers < 1:
        raise ValueError("workers must be >= 1")

    paths = ProcessedDatasetPaths.from_dataset_root(processed_root=processed_root, dataset_root=dataset_root)
    _LOGGER.info("Parsing dataset from %s into %s", dataset_root, paths.root)
    rows = _parse_scores(
        dataset_root,
        paths=paths,
        overwrite=overwrite,
        workers=workers,
        show_progress=show_progress,
        profiler=profiler,
    )
    _write_manifest(rows, paths=paths, profiler=profiler)
    return _parse_dataset_result(rows, paths=paths)


def _parse_scores(
    dataset_root: Path,
    *,
    paths: ProcessedDatasetPaths,
    overwrite: bool,
    workers: int,
    show_progress: bool,
    profiler: ProfilerProtocol,
) -> list[dict[str, object]]:
    return _process_parsed_scores(
        dataset_root,
        paths=paths,
        overwrite=overwrite,
        workers=workers,
        show_progress=show_progress,
        profiler=profiler,
    )


def _process_parsed_scores(
    dataset_root: Path,
    *,
    paths: ProcessedDatasetPaths,
    overwrite: bool,
    workers: int,
    show_progress: bool,
    profiler: ProfilerProtocol,
) -> list[dict[str, object]]:
    _LOGGER.info("Collecting MusicXML files from %s", dataset_root)
    started_at = perf_counter()
    with profiler.measure("collect_musicxml_files"):
        source_paths = collect_musicxml_files(dataset_root)
    _LOGGER.info("Collected %s MusicXML file(s) in %.1fs", len(source_paths), perf_counter() - started_at)
    _LOGGER.info("Building parse plan")
    started_at = perf_counter()
    with profiler.measure("build_parse_plan"):
        plan, reuse_stats = _build_parse_plan(
            source_paths,
            dataset_root=dataset_root,
            paths=paths,
            overwrite=overwrite,
        )
    _LOGGER.info("Built parse plan in %.1fs", perf_counter() - started_at)
    _log_reuse_stats(source_count=len(source_paths), task_count=len(plan.tasks), stats=reuse_stats)
    _LOGGER.info("Running parse tasks: tasks=%s workers=%s", len(plan.tasks), workers)
    started_at = perf_counter()
    with profiler.measure("run_parse_tasks"):
        _run_tasks_with_partial_manifest(
            plan,
            paths=paths,
            workers=workers,
            show_progress=show_progress,
            profiler=profiler,
        )
    _LOGGER.info("Finished parse tasks in %.1fs", perf_counter() - started_at)
    _LOGGER.info("Finalizing parse results")
    started_at = perf_counter()
    with profiler.measure("finalize_parse_results"):
        results = _filled_results(plan.ordered_results)
        rows = [result.row for result in results]
        parsed_count = sum(row[ParsedManifestField.STATUS] == ParsedManifestStatus.SUCCESS.value for row in rows)
    _LOGGER.info("Finalized parse results in %.1fs", perf_counter() - started_at)
    _LOGGER.info("Parsed %s/%s source file(s)", parsed_count, len(source_paths))
    return rows


def _build_parse_plan(
    source_paths: list[Path],
    *,
    dataset_root: Path,
    paths: ProcessedDatasetPaths,
    overwrite: bool,
) -> tuple[_ParsePlan, _ReuseStats]:
    reusable_rows = reusable_parsed_manifest_rows(paths=paths, overwrite=overwrite)
    ordered_results: list[ParsedScoreResult | None] = [None] * len(source_paths)
    tasks: list[ParsedScoreTask] = []
    success_count = 0
    error_count = 0
    issues: dict[ParsedManifestReuseIssue, int] = {}

    for index, source_path in enumerate(source_paths):
        reused_result, reuse_issue = _reuse_or_task(
            index=index,
            source_path=source_path,
            dataset_root=dataset_root,
            paths=paths,
            overwrite=overwrite,
            reusable_rows=reusable_rows,
            tasks=tasks,
        )
        if reuse_issue is not None:
            issues[reuse_issue] = issues.get(reuse_issue, 0) + 1

        if reused_result is None:
            continue

        ordered_results[index] = reused_result
        if reused_result.row[ParsedManifestField.STATUS] == ParsedManifestStatus.ERROR.value:
            error_count += 1
        else:
            success_count += 1

    return (
        _ParsePlan(ordered_results=ordered_results, tasks=tasks),
        _ReuseStats(success_count=success_count, error_count=error_count, issues=issues),
    )


def _reuse_or_task(
    *,
    index: int,
    source_path: Path,
    dataset_root: Path,
    paths: ProcessedDatasetPaths,
    overwrite: bool,
    reusable_rows: dict[str, dict[str, str]],
    tasks: list[ParsedScoreTask],
) -> tuple[ParsedScoreResult | None, ParsedManifestReuseIssue | None]:
    source_id_value = source_id(source_path, dataset_root=dataset_root)
    reused_result, reuse_issue = reused_parsed_result(
        index=index,
        source_id_value=source_id_value,
        source_path=source_path,
        paths=paths,
        row=reusable_rows.get(source_id_value),
    )

    if reused_result is None:
        tasks.append(
            ParsedScoreTask(
                index=index,
                source_path=source_path,
                dataset_root=dataset_root,
                paths=paths,
                overwrite=overwrite,
            )
        )

    return reused_result, reuse_issue


def _log_reuse_stats(
    *,
    source_count: int,
    task_count: int,
    stats: _ReuseStats,
) -> None:
    reused_count = source_count - task_count
    if reused_count:
        _LOGGER.info(
            "Reusing %s parsed manifest row(s): %s success(es), %s error(s)",
            reused_count,
            stats.success_count,
            stats.error_count,
        )

    for issue, count in stats.issues.items():
        _LOGGER.warning("Could not reuse %s parsed manifest row(s): %s", count, issue.value)

    _LOGGER.info("Parsing %s/%s source file(s)", task_count, source_count)


def _run_tasks_with_partial_manifest(
    plan: _ParsePlan,
    *,
    paths: ProcessedDatasetPaths,
    workers: int,
    show_progress: bool,
    profiler: ProfilerProtocol,
) -> None:
    completed = False
    try:
        run_parsed_score_tasks(
            plan.tasks,
            workers=workers,
            show_progress=show_progress,
            ordered_results=plan.ordered_results,
            profiler=profiler,
        )
        completed = True
    finally:
        if not completed:
            _write_partial_manifest(plan.ordered_results, paths=paths, profiler=profiler)


def _write_partial_manifest(
    results: list[ParsedScoreResult | None],
    *,
    paths: ProcessedDatasetPaths,
    profiler: ProfilerProtocol,
) -> None:
    partial_rows = [result.row for result in results if result is not None]
    if not partial_rows:
        return

    _LOGGER.info("Writing partial parsed manifest to %s", paths.parsed_manifest_path)
    with profiler.measure("write_partial_parsed_manifest"):
        write_parsed_manifest(partial_rows, paths.parsed_manifest_path)
    _LOGGER.info("Wrote partial parsed manifest with %s row(s)", len(partial_rows))


def _write_manifest(
    rows: list[dict[str, object]],
    *,
    paths: ProcessedDatasetPaths,
    profiler: ProfilerProtocol,
) -> None:
    _LOGGER.info("Writing parsed manifest to %s", paths.parsed_manifest_path)
    with profiler.measure("write_parsed_manifest"):
        write_parsed_manifest(rows, paths.parsed_manifest_path)

    _LOGGER.info("Wrote parsed manifest: %s", paths.parsed_manifest_path)


def _parse_dataset_result(
    rows: list[dict[str, object]],
    *,
    paths: ProcessedDatasetPaths,
) -> ParseDatasetResult:
    return ParseDatasetResult(
        parsed_manifest_path=paths.parsed_manifest_path,
        parsed_count=sum(1 for row in rows if row[ParsedManifestField.STATUS] == ParsedManifestStatus.SUCCESS.value),
        error_count=sum(1 for row in rows if row[ParsedManifestField.STATUS] == ParsedManifestStatus.ERROR.value),
    )


def _filled_results(results: list[ParsedScoreResult | None]) -> list[ParsedScoreResult]:
    return [result for result in results if result is not None]
