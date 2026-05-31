from pathlib import Path

from musak_model.processing.profiler.torch_profiler import TorchProfilerBackend
from musak_shared.profiling import NULL_PROFILER, CProfileBackend, NullProfiler, Profiler


def build_processing_profiler(
    *,
    enabled: bool,
    output_directory: Path,
) -> Profiler | NullProfiler:
    if not enabled:
        return NULL_PROFILER

    return Profiler(
        output_directory=output_directory,
        backends=(CProfileBackend(), TorchProfilerBackend()),
        retain_records=False,
    )
