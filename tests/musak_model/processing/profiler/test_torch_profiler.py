from pathlib import Path

from musak_model.processing.profiler import TorchProfilerBackend
from musak_shared.profiling import Profiler


def test_torch_backend_writes_torch_reports(tmp_path: Path) -> None:
    with Profiler(output_directory=tmp_path, backends=(TorchProfilerBackend(),), retain_records=False) as profiler:
        with profiler.measure("scale_match", source_file=Path("1/0001.mxl")):
            sum(index * index for index in range(1000))

    profiler.write_reports()

    assert (tmp_path / "torch_profiler_table.txt").exists()
    assert (tmp_path / "torch_profiler_functions.csv").exists()
    assert (tmp_path / "torch_profiler_trace.json").exists()
