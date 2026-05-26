from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sqlite3
from collections import Counter
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Final

from musak_model.analysis.n_grams.config import NGramAnalysisConfig
from musak_model.analysis.n_grams.figure.builder import scale_size_for_type
from musak_model.analysis.n_grams.figure.parser import extract_hand_onset_runs
from musak_model.analysis.n_grams.figure.signature import (
    figure_signature_chords_only,
    figure_signature_from_json,
    figure_signature_in_scale,
    figure_signature_monophonic,
    figure_signature_to_json,
    figure_signature_to_ngram,
    iter_figure_signatures_from_run,
)
from musak_model.analysis.n_grams.profile.artifacts import FigureArtifactPaths
from musak_model.analysis.n_grams.profile.io import COUNT_CSV_COLUMNS
from musak_model.analysis.n_grams.profile.schema import FigureProfile, FigureProfileGroup, FigureProfileMetadata
from musak_model.processing.paths import ENCODED_JSONL_NAME
from musak_model.processing.progress import progress
from musak_model.processing.snapshot import TokenizerSnapshot
from musak_model.processing.workers import process_pool_context
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand, ScaleType
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.schema import EncodedExercise

type FigureCountKey = tuple[str, str, int, str]
type FigureCountCounter = Counter[FigureCountKey]
type FigureGroupKey = tuple[str, str, int]
type FigureGroupTotals = dict[FigureGroupKey, list[int]]

_STATE_VERSION: Final[int] = 1
_WORK_DATABASE_NAME: Final[str] = "work.sqlite3"
_METADATA_STATE_KEY: Final[str] = "state_key"
_METADATA_ENCODED_SAMPLE_COUNT: Final[str] = "encoded_sample_count"
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FigureBatchTask:
    batch_index: int
    sample_start_index: int
    encoded_lines: tuple[str, ...]
    tokenization_config: TokenizationConfig
    min_n: int
    max_n: int


@dataclass(frozen=True)
class FigureBatchResult:
    batch_index: int
    sample_start_index: int
    encoded_sample_count: int
    counts: FigureCountCounter
    sample_payloads: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class FigureStoreSummary:
    encoded_sample_count: int
    profile_group_count: int
    sample_profile_count: int


def extract_streaming_figure_artifacts(
    *,
    encoded_directory: Path,
    artifact_paths: FigureArtifactPaths,
    config: NGramAnalysisConfig,
    snapshot: TokenizerSnapshot,
    output_path: Path | None,
    analysis_config_path: Path,
    show_progress: bool,
    overwrite: bool,
    resume: bool,
) -> FigureStoreSummary:
    store_path = figure_work_store_path(artifact_paths)
    state_key = figure_state_key(config=config, snapshot=snapshot)
    if overwrite:
        _LOGGER.info("Clearing existing figure artifacts before extraction: %s", artifact_paths.root_directory)
        clear_figure_work(artifact_paths)

    if complete_figure_artifacts_exist(artifact_paths) and not overwrite:
        _LOGGER.info("Reusing complete figure artifacts: %s", artifact_paths.root_directory)
        return existing_figure_summary(artifact_paths)

    _LOGGER.info("Opening figure work store: %s", store_path)
    started_at = perf_counter()
    with FigureWorkStore(store_path, state_key=state_key, resume=resume) as store:
        _LOGGER.info("Opened figure work store in %.1fs", perf_counter() - started_at)
        process_missing_batches(
            store,
            encoded_jsonl_path=encoded_directory / ENCODED_JSONL_NAME,
            tokenization_config=TokenizationConfig.model_validate(snapshot.tokenization_config),
            config=config,
            show_progress=show_progress,
        )
        _LOGGER.info("Exporting figure artifacts")
        started_at = perf_counter()
        summary = export_figure_artifacts(
            store,
            artifact_paths=artifact_paths,
            output_path=output_path,
            analysis_config_path=analysis_config_path,
            min_n=config.min_n,
            max_n=config.max_n,
            limit_per_group=config.limit_per_group,
        )
        _LOGGER.info("Exported figure artifacts in %.1fs", perf_counter() - started_at)

    store_path.unlink(missing_ok=True)
    _LOGGER.info("Removed completed figure work store: %s", store_path)
    return summary


def process_figure_batch_task(task: FigureBatchTask) -> FigureBatchResult:
    duration_vocabulary = DurationVocabulary(task.tokenization_config)
    token_vocabulary = TokenVocabulary(duration_vocabulary)
    counts: FigureCountCounter = Counter()
    sample_payloads: list[tuple[int, str]] = []
    for sample_offset, line in enumerate(task.encoded_lines):
        sample_index = task.sample_start_index + sample_offset
        sample = EncodedExercise.model_validate_json(line)
        sample_counts = count_sample_figure_signatures(
            sample,
            duration_vocabulary=duration_vocabulary,
            token_vocabulary=token_vocabulary,
            min_n=task.min_n,
            max_n=task.max_n,
        )
        counts.update(sample_counts)
        sample_payloads.append((sample_index, sample_profile_payload(sample_index, sample.scale_type, sample_counts)))

    return FigureBatchResult(
        batch_index=task.batch_index,
        sample_start_index=task.sample_start_index,
        encoded_sample_count=len(task.encoded_lines),
        counts=counts,
        sample_payloads=tuple(sample_payloads),
    )


def count_sample_figure_signatures(
    sample: EncodedExercise,
    *,
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
    min_n: int,
    max_n: int,
) -> FigureCountCounter:
    tokens = token_vocabulary.decode(sample.token_ids)
    runs_by_hand = extract_hand_onset_runs(
        tokens,
        duration_vocabulary=duration_vocabulary,
        time_numerator=sample.time_numerator,
        time_denominator=sample.time_denominator,
    )
    scale_size = scale_size_for_type(sample.scale_type)
    counts: FigureCountCounter = Counter()
    for hand, runs in runs_by_hand.items():
        for run in runs:
            for n, signature in iter_figure_signatures_from_run(
                run,
                min_n=min_n,
                max_n=max_n,
                scale_size=scale_size,
            ):
                counts[(sample.scale_type.value, hand.value, n, figure_signature_to_json(signature))] += 1

    return counts


def sample_profile_payload(
    sample_index: int,
    scale_type: ScaleType,
    counts: FigureCountCounter,
) -> str:
    groups: list[FigureProfileGroup] = []
    for (group_scale_type, hand, n), totals in sorted(_figure_group_totals(counts).items()):
        groups.append(
            FigureProfileGroup(
                scale_type=ScaleType(group_scale_type),
                hand=Hand(hand),
                n=n,
                total=totals[0],
                monophonic=totals[1],
                chords_only=totals[2],
                in_scale=totals[3],
            )
        )

    from musak_model.analysis.n_grams.profile.schema import FigureSampleCounts

    return FigureSampleCounts(sample_index=sample_index, scale_type=scale_type, groups=tuple(groups)).model_dump_json()


class FigureWorkStore:
    def __init__(
        self,
        path: Path,
        *,
        state_key: str,
        resume: bool,
    ) -> None:
        self.path = path
        self._state_key = state_key
        self._resume = resume
        self._connection: sqlite3.Connection | None = None

    def __enter__(self) -> FigureWorkStore:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._initialize_schema()
        self._validate_or_initialize_state()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        if self._connection is not None:
            self._connection.close()

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("figure work store is not open")

        return self._connection

    def completed_batch_indexes(self) -> set[int]:
        cursor = self.connection.execute("SELECT batch_index FROM completed_batches")
        return {int(row[0]) for row in cursor.fetchall()}

    def commit_batch(self, result: FigureBatchResult) -> None:
        with self.connection:
            self.connection.executemany(
                """
                INSERT INTO counts(scale_type, hand, n, figure, count)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(scale_type, hand, n, figure)
                DO UPDATE SET count = count + excluded.count
                """,
                (
                    (scale_type, hand, n, figure, count)
                    for (scale_type, hand, n, figure), count in result.counts.items()
                ),
            )
            self.connection.executemany(
                """
                INSERT INTO sample_counts(sample_index, payload)
                VALUES (?, ?)
                ON CONFLICT(sample_index)
                DO UPDATE SET payload = excluded.payload
                """,
                result.sample_payloads,
            )
            self.connection.execute(
                """
                INSERT INTO completed_batches(batch_index, sample_start_index, sample_count)
                VALUES (?, ?, ?)
                ON CONFLICT(batch_index)
                DO UPDATE SET sample_start_index = excluded.sample_start_index, sample_count = excluded.sample_count
                """,
                (result.batch_index, result.sample_start_index, result.encoded_sample_count),
            )
            self._set_metadata(
                _METADATA_ENCODED_SAMPLE_COUNT,
                str(max(self.encoded_sample_count(), result.sample_start_index + result.encoded_sample_count)),
            )

    def encoded_sample_count(self) -> int:
        value = self._metadata_value(_METADATA_ENCODED_SAMPLE_COUNT)
        return int(value) if value is not None else 0

    def _initialize_schema(self) -> None:
        with self.connection:
            self.connection.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS counts(
                    scale_type TEXT NOT NULL,
                    hand TEXT NOT NULL,
                    n INTEGER NOT NULL,
                    figure TEXT NOT NULL,
                    count INTEGER NOT NULL,
                    PRIMARY KEY(scale_type, hand, n, figure)
                )
                """)
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS sample_counts(
                    sample_index INTEGER PRIMARY KEY,
                    payload TEXT NOT NULL
                )
                """)
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS completed_batches(
                    batch_index INTEGER PRIMARY KEY,
                    sample_start_index INTEGER NOT NULL,
                    sample_count INTEGER NOT NULL
                )
                """)

    def _validate_or_initialize_state(self) -> None:
        existing_state_key = self._metadata_value(_METADATA_STATE_KEY)
        if existing_state_key is None:
            with self.connection:
                self._set_metadata(_METADATA_STATE_KEY, self._state_key)
                self._set_metadata(_METADATA_ENCODED_SAMPLE_COUNT, "0")
            return

        if existing_state_key != self._state_key:
            raise RuntimeError("partial figure extraction state does not match the current analysis configuration")

        _ = self._resume

    def _metadata_value(self, key: str) -> str | None:
        cursor = self.connection.execute("SELECT value FROM metadata WHERE key = ?", (key,))
        row = cursor.fetchone()
        return str(row[0]) if row is not None else None

    def _set_metadata(self, key: str, value: str) -> None:
        self.connection.execute(
            """
            INSERT INTO metadata(key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )


def process_missing_batches(
    store: FigureWorkStore,
    *,
    encoded_jsonl_path: Path,
    tokenization_config: TokenizationConfig,
    config: NGramAnalysisConfig,
    show_progress: bool,
) -> None:
    completed_batches = store.completed_batch_indexes()
    _LOGGER.info(
        "Preparing figure n-gram batches: encoded_jsonl=%s completed_batches=%s batch_size=%s workers=%s",
        encoded_jsonl_path,
        len(completed_batches),
        config.batch_size,
        config.workers,
    )
    tasks = figure_batch_tasks(
        encoded_jsonl_path,
        tokenization_config=tokenization_config,
        min_n=config.min_n,
        max_n=config.max_n,
        batch_size=config.batch_size,
        completed_batches=completed_batches,
    )
    if config.workers == 1:
        started_at = perf_counter()
        for task in progress(tasks, description="Counting figure n-gram batches", unit="batch", enabled=show_progress):
            store.commit_batch(process_figure_batch_task(task))
        _LOGGER.info("Finished serial figure n-gram batches in %.1fs", perf_counter() - started_at)
        return

    started_at = perf_counter()
    _process_missing_batches_in_parallel(
        store,
        tasks,
        workers=config.workers,
        show_progress=show_progress,
    )
    _LOGGER.info("Finished parallel figure n-gram batches in %.1fs", perf_counter() - started_at)


def figure_batch_tasks(
    encoded_jsonl_path: Path,
    *,
    tokenization_config: TokenizationConfig,
    min_n: int,
    max_n: int,
    batch_size: int,
    completed_batches: set[int],
) -> Iterator[FigureBatchTask]:
    with encoded_jsonl_path.open("r", encoding="utf-8") as file:
        batch_index = 0
        sample_start_index = 0
        encoded_lines: list[str] = []
        for line in file:
            if line.strip() == "":
                continue

            encoded_lines.append(line)
            if len(encoded_lines) == batch_size:
                if batch_index not in completed_batches:
                    yield FigureBatchTask(
                        batch_index=batch_index,
                        sample_start_index=sample_start_index,
                        encoded_lines=tuple(encoded_lines),
                        tokenization_config=tokenization_config,
                        min_n=min_n,
                        max_n=max_n,
                    )
                batch_index += 1
                sample_start_index += len(encoded_lines)
                encoded_lines.clear()

        if encoded_lines and batch_index not in completed_batches:
            yield FigureBatchTask(
                batch_index=batch_index,
                sample_start_index=sample_start_index,
                encoded_lines=tuple(encoded_lines),
                tokenization_config=tokenization_config,
                min_n=min_n,
                max_n=max_n,
            )


def export_figure_artifacts(
    store: FigureWorkStore,
    *,
    artifact_paths: FigureArtifactPaths,
    output_path: Path | None,
    analysis_config_path: Path,
    min_n: int,
    max_n: int,
    limit_per_group: int | None,
) -> FigureStoreSummary:
    profile = profile_from_store(store, min_n=min_n, max_n=max_n)
    sample_profile_count = export_sample_counts(store, artifact_paths.by_sample_path)
    export_counts_csv(store, artifact_paths.counts_path, limit_per_group=None)
    write_profile_atomically(profile, artifact_paths.profile_path)
    copy_file_atomically(analysis_config_path, artifact_paths.config_path)
    if output_path is not None:
        export_counts_csv(store, output_path, limit_per_group=limit_per_group)

    return FigureStoreSummary(
        encoded_sample_count=store.encoded_sample_count(),
        profile_group_count=len(profile.groups),
        sample_profile_count=sample_profile_count,
    )


def profile_from_store(
    store: FigureWorkStore,
    *,
    min_n: int,
    max_n: int,
) -> FigureProfile:
    groups: list[FigureProfileGroup] = []
    for (scale_type, hand, n), totals in sorted(_figure_group_totals(_iter_store_counts(store)).items()):
        groups.append(
            FigureProfileGroup(
                scale_type=ScaleType(scale_type),
                hand=Hand(hand),
                n=n,
                total=totals[0],
                monophonic=totals[1],
                chords_only=totals[2],
                in_scale=totals[3],
            )
        )

    return FigureProfile(
        metadata=FigureProfileMetadata(min_n=min_n, max_n=max_n, sample_count=store.encoded_sample_count()),
        groups=tuple(groups),
    )


def export_counts_csv(
    store: FigureWorkStore,
    path: Path,
    *,
    limit_per_group: int | None,
) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COUNT_CSV_COLUMNS)
        writer.writeheader()
        for scale_type, hand, n, figure_json, count in _iter_limited_store_rows(store, limit_per_group=limit_per_group):
            writer.writerow(
                {
                    "scale_type": scale_type,
                    "hand": hand,
                    "n": n,
                    "count": count,
                    "figure": figure_signature_to_ngram(figure_signature_from_json(figure_json)).model_dump_json(),
                }
            )
    temp_path.replace(path)


def export_sample_counts(
    store: FigureWorkStore,
    path: Path,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    count = 0
    with temp_path.open("w", encoding="utf-8") as file:
        cursor = store.connection.execute("SELECT payload FROM sample_counts ORDER BY sample_index")
        for (payload,) in cursor:
            file.write(str(payload))
            file.write("\n")
            count += 1
    temp_path.replace(path)
    return count


def write_profile_atomically(profile: FigureProfile, path: Path) -> None:
    from musak_model.processing.io import JSON_INDENT

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(profile.model_dump_json(indent=JSON_INDENT), encoding="utf-8")
    temp_path.replace(path)


def copy_file_atomically(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() == target_path.resolve():
        return

    temp_path = target_path.with_suffix(f"{target_path.suffix}.tmp")
    shutil.copyfile(source_path, temp_path)
    temp_path.replace(target_path)


def figure_work_store_path(artifact_paths: FigureArtifactPaths) -> Path:
    return artifact_paths.root_directory / _WORK_DATABASE_NAME


def clear_figure_work(artifact_paths: FigureArtifactPaths) -> None:
    for path in (
        artifact_paths.config_path,
        artifact_paths.counts_path,
        artifact_paths.profile_path,
        artifact_paths.by_sample_path,
        figure_work_store_path(artifact_paths),
    ):
        path.unlink(missing_ok=True)


def complete_figure_artifacts_exist(artifact_paths: FigureArtifactPaths) -> bool:
    return all(
        path.exists()
        for path in (
            artifact_paths.config_path,
            artifact_paths.counts_path,
            artifact_paths.profile_path,
            artifact_paths.by_sample_path,
        )
    )


def existing_figure_summary(artifact_paths: FigureArtifactPaths) -> FigureStoreSummary:
    from musak_model.analysis.n_grams.profile.io import read_figure_profile

    profile = read_figure_profile(artifact_paths.profile_path)
    sample_profile_count = sum(
        1 for line in artifact_paths.by_sample_path.read_text(encoding="utf-8").splitlines() if line
    )
    return FigureStoreSummary(
        encoded_sample_count=profile.metadata.sample_count,
        profile_group_count=len(profile.groups),
        sample_profile_count=sample_profile_count,
    )


def figure_state_key(
    *,
    config: NGramAnalysisConfig,
    snapshot: TokenizerSnapshot,
) -> str:
    payload = {
        "version": _STATE_VERSION,
        "tokenizer_hash": snapshot.tokenizer_hash,
        "min_n": config.min_n,
        "max_n": config.max_n,
        "batch_size": config.batch_size,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _process_missing_batches_in_parallel(
    store: FigureWorkStore,
    tasks: Iterator[FigureBatchTask],
    *,
    workers: int,
    show_progress: bool,
) -> None:
    _LOGGER.info("Running parallel figure n-gram batches: workers=%s", workers)
    with ProcessPoolExecutor(max_workers=workers, mp_context=process_pool_context()) as executor:
        pending: dict[Future[FigureBatchResult], int] = {}
        task_iterator = iter(tasks)
        _submit_pending_tasks(executor, pending, task_iterator, workers=workers)
        completed_count = 0
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                store.commit_batch(result)
                completed_count += 1
                del pending[future]
            _submit_pending_tasks(executor, pending, task_iterator, workers=workers)
            if show_progress and completed_count % 10 == 0:
                _LOGGER.info("Completed %s figure n-gram batch(es)", completed_count)
        _LOGGER.info("Completed %s figure n-gram batch(es)", completed_count)


def _submit_pending_tasks(
    executor: ProcessPoolExecutor,
    pending: dict[Future[FigureBatchResult], int],
    tasks: Iterator[FigureBatchTask],
    *,
    workers: int,
) -> None:
    max_pending = workers * 2
    while len(pending) < max_pending:
        try:
            task = next(tasks)
        except StopIteration:
            return

        pending[executor.submit(process_figure_batch_task, task)] = task.batch_index


def _figure_group_totals(counts: Iterable[tuple[FigureCountKey, int]] | FigureCountCounter) -> FigureGroupTotals:
    items = counts.items() if isinstance(counts, Counter) else counts
    totals_by_group: FigureGroupTotals = {}
    for (scale_type, hand, n, figure_json), count in items:
        signature = figure_signature_from_json(figure_json)
        totals = totals_by_group.setdefault((scale_type, hand, n), [0, 0, 0, 0])
        totals[0] += count
        if figure_signature_monophonic(signature):
            totals[1] += count
        if figure_signature_chords_only(signature):
            totals[2] += count
        if figure_signature_in_scale(signature):
            totals[3] += count

    return totals_by_group


def _iter_store_counts(store: FigureWorkStore) -> Iterator[tuple[FigureCountKey, int]]:
    cursor = store.connection.execute("SELECT scale_type, hand, n, figure, count FROM counts")
    for scale_type, hand, n, figure, count in cursor:
        yield (str(scale_type), str(hand), int(n), str(figure)), int(count)


def _iter_limited_store_rows(
    store: FigureWorkStore,
    *,
    limit_per_group: int | None,
) -> Iterator[tuple[str, str, int, str, int]]:
    query = """
        SELECT scale_type, hand, n, figure, count
        FROM counts
        ORDER BY scale_type, hand, n, count DESC, figure
    """
    current_group: tuple[str, str, int] | None = None
    current_group_count = 0
    cursor = store.connection.execute(query)
    for scale_type, hand, n, figure, count in cursor:
        group = (str(scale_type), str(hand), int(n))
        if group != current_group:
            current_group = group
            current_group_count = 0

        if limit_per_group is not None and current_group_count >= limit_per_group:
            continue

        current_group_count += 1
        yield str(scale_type), str(hand), int(n), str(figure), int(count)
