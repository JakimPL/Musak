from __future__ import annotations

import cProfile
import csv
import time
from contextlib import AbstractContextManager, ExitStack, nullcontext
from dataclasses import asdict
from pathlib import Path
from types import TracebackType
from typing import TextIO

import torch

from musak_model.processing.profiler.cpu import create_cpu_profiler, write_cpu_reports
from musak_model.processing.profiler.reports import (
    write_records_report,
    write_source_stats_report,
    write_stage_stats_report,
    write_summary_report,
)
from musak_model.processing.profiler.schema import (
    ProcessingProfileRecord,
    ProcessingProfileStageStats,
    ProcessingProfileSummary,
)
from musak_model.processing.profiler.torch_profiler import (
    create_torch_profiler,
    synchronize_cuda,
    write_torch_reports,
)


class ProcessingProfiler:
    def __init__(
        self,
        *,
        output_dir: Path,
        use_torch_profiler_labels: bool = True,
        synchronize_cuda_before_measurement: bool = True,
        retain_records: bool = True,
    ) -> None:
        self._output_dir = output_dir
        self._records: list[ProcessingProfileRecord] = []
        self._retain_records = retain_records
        self._use_torch_profiler_labels = use_torch_profiler_labels
        self._synchronize_cuda_before_measurement = synchronize_cuda_before_measurement
        self._cpu_profiler: cProfile.Profile | None = None
        self._torch_profiler: torch.profiler.profile | None = None
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
    def records(self) -> tuple[ProcessingProfileRecord, ...]:
        return tuple(self._records)

    def __enter__(self) -> ProcessingProfiler:
        if not self._retain_records:
            self._open_records_report()
        self._cpu_profiler = create_cpu_profiler()
        self._cpu_profiler.enable()
        self._torch_profiler = create_torch_profiler()
        self._torch_profiler.__enter__()  # type: ignore[no-untyped-call]
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._cpu_profiler is not None:
            self._cpu_profiler.disable()
        if self._torch_profiler is not None:
            self._torch_profiler.__exit__(exc_type, exc_value, traceback)  # type: ignore[no-untyped-call]

        self._close_records_report()
        return None

    def measure(self, stage: str, *, source_file: Path | None = None) -> AbstractContextManager[None]:
        return _ProcessingProfileScope(
            profiler=self,
            stage=stage,
            source_file=source_file.as_posix() if source_file is not None else "",
        )

    def record(self, *, stage: str, source_file: str, seconds: float) -> None:
        record = ProcessingProfileRecord(stage=stage, seconds=seconds, source_file=source_file)
        if self._retain_records:
            self._records.append(record)
        else:
            self._write_record(record)
        self._record_summary(record)

    def step(self) -> None:
        return None

    def summary(self) -> ProcessingProfileSummary:
        return ProcessingProfileSummary(
            total_seconds=self._total_seconds,
            stage_totals=self._stage_totals,
            stage_counts=self._stage_counts,
            stage_means={stage: self._stage_totals[stage] / count for stage, count in self._stage_counts.items()},
            per_file_totals=self._per_file_totals,
        )

    def write_reports(self) -> None:
        self._close_records_report()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        write_summary_report(self.summary(), self._output_dir)
        write_stage_stats_report(self.stage_stats(), self._output_dir)
        write_source_stats_report(self._per_file_totals, self._output_dir)
        if self._retain_records:
            write_records_report(self._records, self._output_dir)
        if self._cpu_profiler is not None:
            write_cpu_reports(self._cpu_profiler, self._output_dir)
        if self._torch_profiler is not None:
            write_torch_reports(self._torch_profiler, self._output_dir)

    def stage_stats(self) -> list[ProcessingProfileStageStats]:
        return [
            ProcessingProfileStageStats(
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

    def _record_summary(self, record: ProcessingProfileRecord) -> None:
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
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._records_file = (self._output_dir / "records.csv").open("w", encoding="utf-8", newline="")
        self._records_writer = csv.DictWriter(
            self._records_file,
            fieldnames=list(ProcessingProfileRecord.__dataclass_fields__),
        )
        self._records_writer.writeheader()

    def _write_record(self, record: ProcessingProfileRecord) -> None:
        if self._records_writer is None:
            self._open_records_report()
        if self._records_writer is None:
            raise RuntimeError("processing profile records writer is not open")

        self._records_writer.writerow(asdict(record))

    def _close_records_report(self) -> None:
        if self._records_file is not None:
            self._records_file.close()
            self._records_file = None
            self._records_writer = None


class _ProcessingProfileScope(AbstractContextManager[None]):
    def __init__(self, *, profiler: ProcessingProfiler, stage: str, source_file: str) -> None:
        self._profiler = profiler
        self._stage = stage
        self._source_file = source_file
        self._started = 0.0
        self._stack = ExitStack()

    def __enter__(self) -> None:
        if self._profiler._use_torch_profiler_labels:
            self._stack.enter_context(torch.profiler.record_function(self._stage))
        else:
            self._stack.enter_context(nullcontext())

        synchronize_cuda(self._profiler._synchronize_cuda_before_measurement)
        self._started = time.perf_counter()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        synchronize_cuda(self._profiler._synchronize_cuda_before_measurement)
        seconds = time.perf_counter() - self._started
        self._profiler.record(stage=self._stage, source_file=self._source_file, seconds=seconds)
        return self._stack.__exit__(exc_type, exc_value, traceback)
