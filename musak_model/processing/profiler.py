import csv
import json
import statistics
import time
from contextlib import AbstractContextManager, ExitStack, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self

import torch


@dataclass(frozen=True)
class ProcessingProfileRecord:
    stage: str
    seconds: float
    source_file: str


@dataclass(frozen=True)
class ProcessingProfileSummary:
    total_seconds: float
    stage_totals: dict[str, float]
    stage_counts: dict[str, int]
    stage_means: dict[str, float]
    per_file_totals: dict[str, float]


class ProcessingProfilerProtocol(Protocol):
    @property
    def enabled(self) -> bool: ...

    def measure(self, stage: str, *, source_file: Path | None = None) -> AbstractContextManager[None]: ...

    def step(self) -> None: ...

    def write_reports(self) -> None: ...


class ProcessingProfiler:
    def __init__(
        self,
        *,
        output_dir: Path,
        use_torch_profiler_labels: bool = True,
        synchronize_cuda: bool = True,
    ) -> None:
        self._output_dir = output_dir
        self._records: list[ProcessingProfileRecord] = []
        self._use_torch_profiler_labels = use_torch_profiler_labels
        self._synchronize_cuda = synchronize_cuda
        self._torch_profiler: torch.profiler.profile | None = None

    @property
    def enabled(self) -> bool:
        return True

    @property
    def records(self) -> tuple[ProcessingProfileRecord, ...]:
        return tuple(self._records)

    def __enter__(self) -> Self:
        self._torch_profiler = _torch_profiler()
        self._torch_profiler.__enter__()  # type: ignore[no-untyped-call]
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
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

    def record(
        self,
        *,
        stage: str,
        source_file: str,
        seconds: float,
    ) -> None:
        self._records.append(
            ProcessingProfileRecord(
                stage=stage,
                seconds=seconds,
                source_file=source_file,
            )
        )

    def step(self) -> None:
        if self._torch_profiler is not None:
            self._torch_profiler.step()

    def summary(self) -> ProcessingProfileSummary:
        stage_values: dict[str, list[float]] = {}
        per_file_totals: dict[str, float] = {}
        for record in self._records:
            stage_values.setdefault(record.stage, []).append(record.seconds)
            if record.source_file != "":
                per_file_totals[record.source_file] = per_file_totals.get(record.source_file, 0.0) + record.seconds

        stage_totals = {stage: sum(values) for stage, values in stage_values.items()}
        return ProcessingProfileSummary(
            total_seconds=sum(record.seconds for record in self._records),
            stage_totals=stage_totals,
            stage_counts={stage: len(values) for stage, values in stage_values.items()},
            stage_means={stage: statistics.fmean(values) for stage, values in stage_values.items()},
            per_file_totals=per_file_totals,
        )

    def write_reports(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        summary = self.summary()
        (self._output_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")
        with (self._output_dir / "records.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(ProcessingProfileRecord.__dataclass_fields__))
            writer.writeheader()
            for record in self._records:
                writer.writerow(asdict(record))

        if self._torch_profiler is None:
            return

        sort_by = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
        (self._output_dir / "torch_profiler_table.txt").write_text(
            self._torch_profiler.key_averages().table(sort_by=sort_by, row_limit=80),
            encoding="utf-8",
        )
        self._torch_profiler.export_chrome_trace(str(self._output_dir / "torch_profiler_trace.json"))


class NullProcessingProfiler:
    @property
    def enabled(self) -> bool:
        return False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def measure(self, stage: str, *, source_file: Path | None = None) -> AbstractContextManager[None]:
        return nullcontext()

    def step(self) -> None:
        return None

    def write_reports(self) -> None:
        return None


NULL_PROCESSING_PROFILER = NullProcessingProfiler()


def build_processing_profiler(
    *,
    enabled: bool,
    output_dir: Path,
) -> ProcessingProfiler | NullProcessingProfiler:
    if not enabled:
        return NULL_PROCESSING_PROFILER

    return ProcessingProfiler(output_dir=output_dir)


class _ProcessingProfileScope(AbstractContextManager[None]):
    def __init__(
        self,
        *,
        profiler: ProcessingProfiler,
        stage: str,
        source_file: str,
    ) -> None:
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

        _synchronize_cuda(self._profiler._synchronize_cuda)
        self._started = time.perf_counter()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        _synchronize_cuda(self._profiler._synchronize_cuda)
        seconds = time.perf_counter() - self._started
        self._profiler.record(stage=self._stage, source_file=self._source_file, seconds=seconds)
        return self._stack.__exit__(exc_type, exc_value, traceback)


def _torch_profiler() -> torch.profiler.profile:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    return torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )


def _synchronize_cuda(enabled: bool) -> None:
    if enabled and torch.cuda.is_available():
        torch.cuda.synchronize()
