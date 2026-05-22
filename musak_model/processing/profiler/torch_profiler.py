from __future__ import annotations

from pathlib import Path

import torch


def create_torch_profiler() -> torch.profiler.profile:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    return torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )


def synchronize_cuda(enabled: bool) -> None:
    if enabled and torch.cuda.is_available():
        torch.cuda.synchronize()


def write_torch_reports(profiler: torch.profiler.profile, output_dir: Path) -> None:
    sort_by = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
    (output_dir / "torch_profiler_table.txt").write_text(
        profiler.key_averages().table(sort_by=sort_by, row_limit=80),
        encoding="utf-8",
    )
    profiler.export_chrome_trace(str(output_dir / "torch_profiler_trace.json"))
