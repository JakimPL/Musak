from __future__ import annotations

import csv
import time
from contextlib import AbstractContextManager, ExitStack
from dataclasses import asdict
from pathlib import Path
from types import TracebackType
from typing import Self, TextIO

from musak_shared.profiling.protocol import ProfilerBackend
from musak_shared.profiling.reports import (
    write_records_report,
    write_source_stats_report,
    write_stage_stats_report,
    write_summary_report,
)
from musak_shared.profiling.schema import ProfileRecord, ProfileStageStats, ProfileSummary


class Profiler:
    def __init__(
        self,
        *,
        output_directory: Path,
        backends: tuple[ProfilerBackend, ...],
        retain_records: bool = True,
    ) -> None:
        self._output_directory = output_directory
        self._backends = backends
        self._retain_records = retain_records
        self._records: list[ProfileRecord] = []
        self._session_stack: ExitStack | None = None
        self._records_file: TextIO | None = None
        self._records_writer: csv.DictWriter[str] | None = None
        self._total_seconds = 0.0
        self._stage_totals: dict[str, float] = {}
        self._stage_counts: dict[str, int] = {}
        self._stage_minimums: dict[str, float] = {}
        self._stage_maximums: dict[str, float] = {}
        self._per_file_totals: dict[str, float] = {}

    @property
    def enabled(self) -> bool:
        return True

    @property
    def records(self) -> tuple[ProfileRecord, ...]:
        return tuple(self._records)

    def __enter__(self) -> Self:
        if not self._retain_records:
            self._open_records_report()
        self._session_stack = ExitStack()
        for backend in self._backends:
            self._session_stack.enter_context(backend.session())
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._session_stack is not None:
            self._session_stack.close()
            self._session_stack = None
        self._close_records_report()

    def measure(self, stage: str, *, source_file: Path | None = None) -> AbstractContextManager[None]:
        return _ProfileScope(
            profiler=self,
            backends=self._backends,
            stage=stage,
            source_file=source_file.as_posix() if source_file is not None else "",
        )

    def record(self, *, stage: str, source_file: str, seconds: float) -> None:
        record = ProfileRecord(stage=stage, seconds=seconds, source_file=source_file)
        if self._retain_records:
            self._records.append(record)
        else:
            self._write_record(record)
        self._record_summary(record)

    def step(self) -> None:
        return None

    def summary(self) -> ProfileSummary:
        return ProfileSummary(
            total_seconds=self._total_seconds,
            stage_totals=self._stage_totals,
            stage_counts=self._stage_counts,
            stage_means={stage: self._stage_totals[stage] / count for stage, count in self._stage_counts.items()},
            per_file_totals=self._per_file_totals,
        )

    def write_reports(self) -> None:
        self._close_records_report()
        self._output_directory.mkdir(parents=True, exist_ok=True)
        write_summary_report(self.summary(), self._output_directory)
        write_stage_stats_report(self.stage_stats(), self._output_directory)
        write_source_stats_report(self._per_file_totals, self._output_directory)
        if self._retain_records:
            write_records_report(self._records, self._output_directory)
        for backend in self._backends:
            backend.write_reports(self._output_directory)

    def stage_stats(self) -> list[ProfileStageStats]:
        return [
            ProfileStageStats(
                stage=stage,
                count=count,
                total_seconds=self._stage_totals[stage],
                mean_seconds=self._stage_totals[stage] / count,
                min_seconds=self._stage_minimums[stage],
                max_seconds=self._stage_maximums[stage],
            )
            for stage, count in self._sorted_stage_counts()
        ]

    def _sorted_stage_counts(self) -> list[tuple[str, int]]:
        return sorted(self._stage_counts.items(), key=lambda item: self._stage_totals[item[0]], reverse=True)

    def _record_summary(self, record: ProfileRecord) -> None:
        self._total_seconds += record.seconds
        self._stage_totals[record.stage] = self._stage_totals.get(record.stage, 0.0) + record.seconds
        self._stage_counts[record.stage] = self._stage_counts.get(record.stage, 0) + 1
        self._stage_minimums[record.stage] = min(record.seconds, self._stage_minimums.get(record.stage, record.seconds))
        self._stage_maximums[record.stage] = max(record.seconds, self._stage_maximums.get(record.stage, record.seconds))
        if record.source_file != "":
            self._per_file_totals[record.source_file] = (
                self._per_file_totals.get(record.source_file, 0.0) + record.seconds
            )

    def _open_records_report(self) -> None:
        self._output_directory.mkdir(parents=True, exist_ok=True)
        self._records_file = (self._output_directory / "records.csv").open("w", encoding="utf-8", newline="")
        self._records_writer = csv.DictWriter(
            self._records_file,
            fieldnames=list(ProfileRecord.__dataclass_fields__),
        )
        self._records_writer.writeheader()

    def _write_record(self, record: ProfileRecord) -> None:
        if self._records_writer is None:
            self._open_records_report()
        if self._records_writer is None:
            raise RuntimeError("profiler records writer is not open")

        self._records_writer.writerow(asdict(record))

    def _close_records_report(self) -> None:
        if self._records_file is not None:
            self._records_file.close()
            self._records_file = None
            self._records_writer = None


class _ProfileScope(AbstractContextManager[None]):
    def __init__(
        self,
        *,
        profiler: Profiler,
        backends: tuple[ProfilerBackend, ...],
        stage: str,
        source_file: str,
    ) -> None:
        self._profiler = profiler
        self._backends = backends
        self._stage = stage
        self._source_file = source_file
        self._started = 0.0
        self._stack = ExitStack()

    def __enter__(self) -> None:
        for backend in self._backends:
            self._stack.enter_context(backend.span(self._stage))
        for backend in self._backends:
            backend.before_measure()
        self._started = time.perf_counter()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        for backend in self._backends:
            backend.after_measure()
        seconds = time.perf_counter() - self._started
        self._profiler.record(stage=self._stage, source_file=self._source_file, seconds=seconds)
        return self._stack.__exit__(exc_type, exc_value, traceback)
