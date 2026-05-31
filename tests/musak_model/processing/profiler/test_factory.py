from pathlib import Path

from musak_model.processing.profiler import build_processing_profiler
from musak_shared.profiling import NULL_PROFILER, Profiler


def test_build_processing_profiler_returns_null_when_disabled(tmp_path: Path) -> None:
    assert build_processing_profiler(enabled=False, output_directory=tmp_path) is NULL_PROFILER


def test_build_processing_profiler_composes_a_profiler_when_enabled(tmp_path: Path) -> None:
    profiler = build_processing_profiler(enabled=True, output_directory=tmp_path)

    assert isinstance(profiler, Profiler)
    assert profiler.enabled is True
