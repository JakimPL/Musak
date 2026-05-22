import multiprocessing
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from multiprocessing.context import BaseContext
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile

from music21.exceptions21 import Music21Exception

from musak_model.data.cleaning import clean_parsed_score
from musak_model.data.parser import parse_score
from musak_model.processing.diagnostics import ParseDiagnosticsCapture
from musak_model.processing.ids import source_id
from musak_model.processing.io import load_parsed_score_json, write_json_model
from musak_model.processing.manifest import parsed_error_row, parsed_success_row
from musak_model.processing.parser.schema import ParsedScoreResult, ParsedScoreTask
from musak_model.processing.parser.title import score_title
from musak_model.processing.progress import progress


def run_parsed_score_tasks(
    tasks: list[ParsedScoreTask],
    *,
    workers: int,
    show_progress: bool,
    ordered_results: list[ParsedScoreResult | None],
) -> None:
    if not tasks:
        return

    if workers == 1:
        _run_parsed_score_tasks_serially(tasks, show_progress=show_progress, ordered_results=ordered_results)
        return

    _run_parsed_score_tasks_in_parallel(
        tasks, workers=workers, show_progress=show_progress, ordered_results=ordered_results
    )


def process_parsed_score_task(task: ParsedScoreTask) -> ParsedScoreResult:
    source_id_value = source_id(task.source_path, dataset_root=task.dataset_root)
    parsed_path = task.paths.parsed_score_path(source_id_value)
    title = score_title(task.source_path)
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
        return ParsedScoreResult(
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

    return ParsedScoreResult(
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


def _run_parsed_score_tasks_serially(
    tasks: list[ParsedScoreTask],
    *,
    show_progress: bool,
    ordered_results: list[ParsedScoreResult | None],
) -> None:
    for task in progress(tasks, description="Parsing scores", unit="score", enabled=show_progress):
        result = process_parsed_score_task(task)
        ordered_results[result.index] = result


def _run_parsed_score_tasks_in_parallel(
    tasks: list[ParsedScoreTask],
    *,
    workers: int,
    show_progress: bool,
    ordered_results: list[ParsedScoreResult | None],
) -> None:
    with ProcessPoolExecutor(max_workers=workers, mp_context=_process_pool_context()) as executor:
        futures: dict[Future[ParsedScoreResult], int] = {
            executor.submit(process_parsed_score_task, task): task.index for task in tasks
        }
        completed_futures = as_completed(futures)
        for future in progress(
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
