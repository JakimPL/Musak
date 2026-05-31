from __future__ import annotations

from pathlib import Path

from musak_shared.profiling.collector import Profiler
from musak_shared.profiling.cpu import CProfileBackend
from musak_shared.profiling.null import NULL_PROFILER, NullProfiler


def build_profiler(
    *,
    enabled: bool,
    output_directory: Path,
) -> Profiler | NullProfiler:
    if not enabled:
        return NULL_PROFILER

    return Profiler(
        output_directory=output_directory,
        backends=(CProfileBackend(),),
        retain_records=False,
    )
