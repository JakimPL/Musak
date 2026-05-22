from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Final

import torch

_MICROSECONDS_PER_SECOND: Final[float] = 1_000_000.0
_TORCH_PROFILE_FUNCTIONS_NAME: Final[str] = "torch_profiler_functions.csv"


def create_torch_profiler() -> torch.profiler.profile:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    return torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        acc_events=True,
    )


def synchronize_cuda(enabled: bool) -> None:
    if enabled and torch.cuda.is_available():
        torch.cuda.synchronize()


def write_torch_reports(profiler: torch.profiler.profile, output_dir: Path) -> None:
    sort_by = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
    events = profiler.key_averages()
    (output_dir / "torch_profiler_table.txt").write_text(
        events.table(sort_by=sort_by, row_limit=80),
        encoding="utf-8",
    )
    _write_torch_functions_report(events, output_dir / _TORCH_PROFILE_FUNCTIONS_NAME)
    profiler.export_chrome_trace(str(output_dir / "torch_profiler_trace.json"))


def _write_torch_functions_report(events: Any, path: Path) -> None:
    rows = sorted(
        (_torch_function_row(event) for event in events),
        key=lambda row: (row["self_cuda_seconds"], row["self_cpu_seconds"]),
        reverse=True,
    )
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "operation",
                "calls",
                "total_seconds",
                "self_seconds",
                "self_cpu_seconds",
                "cpu_total_seconds",
                "self_cuda_seconds",
                "cuda_total_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _torch_function_row(event: Any) -> dict[str, str | int | float]:
    self_cpu_seconds = _event_seconds(event, ("self_cpu_time_total",))
    cpu_total_seconds = _event_seconds(event, ("cpu_time_total",))
    self_cuda_seconds = _event_seconds(event, ("self_cuda_time_total", "self_device_time_total"))
    cuda_total_seconds = _event_seconds(event, ("cuda_time_total", "device_time_total"))
    return {
        "operation": str(getattr(event, "key", "")),
        "calls": _event_int(event, "count"),
        "total_seconds": max(cpu_total_seconds, cuda_total_seconds),
        "self_seconds": max(self_cpu_seconds, self_cuda_seconds),
        "self_cpu_seconds": self_cpu_seconds,
        "cpu_total_seconds": cpu_total_seconds,
        "self_cuda_seconds": self_cuda_seconds,
        "cuda_total_seconds": cuda_total_seconds,
    }


def _event_seconds(event: Any, names: tuple[str, ...]) -> float:
    for name in names:
        value = getattr(event, name, None)
        if isinstance(value, int | float):
            return float(value) / _MICROSECONDS_PER_SECOND

    return 0.0


def _event_int(event: Any, name: str) -> int:
    value = getattr(event, name, 0)
    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return int(value)

    return 0
