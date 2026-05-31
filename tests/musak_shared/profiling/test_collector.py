from pathlib import Path

from musak_shared.profiling import CProfileBackend, Profiler


def test_profiler_summarizes_stage_and_file_totals(tmp_path: Path) -> None:
    profiler = Profiler(output_directory=tmp_path, backends=())

    profiler.record(stage="scale_match", source_file="1/0001.mxl", seconds=0.25)
    profiler.record(stage="scale_match", source_file="1/0001.mxl", seconds=0.75)
    profiler.record(stage="encode_segment", source_file="1/0002.mxl", seconds=0.5)

    summary = profiler.summary()

    assert summary.total_seconds == 1.5
    assert summary.stage_totals == {"scale_match": 1.0, "encode_segment": 0.5}
    assert summary.stage_counts == {"scale_match": 2, "encode_segment": 1}
    assert summary.stage_means == {"scale_match": 0.5, "encode_segment": 0.5}
    assert summary.per_file_totals == {"1/0001.mxl": 1.0, "1/0002.mxl": 0.5}
    assert profiler.stage_stats()[0].stage == "scale_match"
    assert profiler.stage_stats()[0].count == 2
    assert profiler.stage_stats()[0].min_seconds == 0.25
    assert profiler.stage_stats()[0].max_seconds == 0.75


def test_profiler_measure_records_a_timed_span(tmp_path: Path) -> None:
    profiler = Profiler(output_directory=tmp_path, backends=())

    with profiler, profiler.measure("scale_match", source_file=Path("1/0001.mxl")):
        pass

    assert [record.stage for record in profiler.records] == ["scale_match"]
    assert profiler.records[0].source_file == "1/0001.mxl"
    assert profiler.records[0].seconds >= 0.0


def test_profiler_writes_reports(tmp_path: Path) -> None:
    profiler = Profiler(output_directory=tmp_path, backends=())
    profiler.record(stage="scale_match", source_file="1/0001.mxl", seconds=0.25)

    profiler.write_reports()

    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "records.csv").exists()
    assert (tmp_path / "stage_stats.csv").exists()
    assert (tmp_path / "source_stats.csv").exists()


def test_cpu_backend_writes_reports_and_streams_records(tmp_path: Path) -> None:
    with Profiler(output_directory=tmp_path, backends=(CProfileBackend(),), retain_records=False) as profiler:
        profiler.record(stage="scale_match", source_file="1/0001.mxl", seconds=0.25)

    profiler.write_reports()

    assert profiler.records == ()
    assert (tmp_path / "records.csv").exists()
    assert (tmp_path / "stage_stats.csv").exists()
    assert (tmp_path / "cpu_profile.pstats").exists()
    assert (tmp_path / "cpu_profile_top.txt").exists()
    assert (tmp_path / "cpu_profile_functions.csv").exists()
