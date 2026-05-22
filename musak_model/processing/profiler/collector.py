from __future__ import annotations

import csv
import json
import statistics
import time
from contextlib import AbstractContextManager, ExitStack, nullcontext
from dataclasses import asdict
from pathlib import Path
from types import TracebackType
from typing import Final

import torch

from musak_model.processing.io import JSON_INDENT
from musak_model.processing.profiler.schema import ProcessingProfileRecord, ProcessingProfileSummary
from musak_model.processing.profiler.torch_profiler import (
    create_torch_profiler,
    synchronize_cuda,
    write_torch_reports,
)

_JSON_INDENT: Final[int] = 4


class ProcessingProfiler:
    def __init__(
        self,
        *,
        output_dir: Path,
        use_torch_profiler_labels: bool = True,
        synchronize_cuda_before_measurement: bool = True,
    ) -> None:
        self._output_dir = output_dir
        self._records: list[ProcessingProfileRecord] = []
        self._use_torch_profiler_labels = use_torch_profiler_labels
        self._synchronize_cuda_before_measurement = synchronize_cuda_before_measurement
        self._torch_profiler: torch.profiler.profile | None = None

    @property
    def enabled(self) -> bool:
        return True

    @property
    def records(self) -> tuple[ProcessingProfileRecord, ...]:
        return tuple(self._records)

    def __enter__(self) -> ProcessingProfiler:
        self._torch_profiler = create_torch_profiler()
        self._torch_profiler.__enter__()  # type: ignore[no-untyped-call]
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._torch_profiler is None:
            return None

        self._torch_profiler.__exit__(exc_type, exc_value, traceback)  # type: ignore[no-untyped-call]
        return None

    def measure(self, stage: str, *, source_file: Path | None = None) -> AbstractContextManager[None]:
        return _ProcessingProfileScope(
            profiler=self,
            stage=stage,
            source_file=source_file.as_posix() if source_file is not None else "",
        )

    def record(self, *, stage: str, source_file: str, seconds: float) -> None:
        self._records.append(ProcessingProfileRecord(stage=stage, seconds=seconds, source_file=source_file))

    def step(self) -> None:
        if self._torch_profiler is not None:
            self._torch_profiler.step()

    def summary(self) -> ProcessingProfileSummary:
        stage_values = _stage_values(self._records)
        stage_totals = {stage: sum(values) for stage, values in stage_values.items()}
        return ProcessingProfileSummary(
            total_seconds=sum(record.seconds for record in self._records),
            stage_totals=stage_totals,
            stage_counts={stage: len(values) for stage, values in stage_values.items()},
            stage_means={stage: statistics.fmean(values) for stage, values in stage_values.items()},
            per_file_totals=_per_file_totals(self._records),
        )

    def write_reports(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        _write_summary_report(self.summary(), self._output_dir)
        _write_records_report(self._records, self._output_dir)
        if self._torch_profiler is not None:
            write_torch_reports(self._torch_profiler, self._output_dir)


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


def _stage_values(records: list[ProcessingProfileRecord]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    for record in records:
        values.setdefault(record.stage, []).append(record.seconds)

    return values


def _per_file_totals(records: list[ProcessingProfileRecord]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for record in records:
        if record.source_file != "":
            totals[record.source_file] = totals.get(record.source_file, 0.0) + record.seconds

    return totals


def _write_summary_report(summary: ProcessingProfileSummary, output_dir: Path) -> None:
    (output_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=JSON_INDENT) + "\n", encoding="utf-8")


def _write_records_report(records: list[ProcessingProfileRecord], output_dir: Path) -> None:
    with (output_dir / "records.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(ProcessingProfileRecord.__dataclass_fields__))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
