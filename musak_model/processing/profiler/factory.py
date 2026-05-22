from __future__ import annotations

from pathlib import Path

from musak_model.processing.profiler.collector import ProcessingProfiler
from musak_model.processing.profiler.null import NULL_PROCESSING_PROFILER, NullProcessingProfiler


def build_processing_profiler(*, enabled: bool, output_dir: Path) -> ProcessingProfiler | NullProcessingProfiler:
    if not enabled:
        return NULL_PROCESSING_PROFILER

    return ProcessingProfiler(output_dir=output_dir, retain_records=False)
