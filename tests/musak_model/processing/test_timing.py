from pathlib import Path

from musak_model.processing.profiler import ProcessingProfiler


def test_processing_profiler_summarizes_stage_and_file_totals(tmp_path: Path) -> None:
    collector = ProcessingProfiler(output_dir=tmp_path)

    collector.record(stage="scale_match", source_file="1/0001.mxl", seconds=0.25)
    collector.record(stage="scale_match", source_file="1/0001.mxl", seconds=0.75)
    collector.record(stage="encode_segment", source_file="1/0002.mxl", seconds=0.5)

    summary = collector.summary()

    assert summary.total_seconds == 1.5
    assert summary.stage_totals == {"scale_match": 1.0, "encode_segment": 0.5}
    assert summary.stage_counts == {"scale_match": 2, "encode_segment": 1}
    assert summary.stage_means == {"scale_match": 0.5, "encode_segment": 0.5}
    assert summary.per_file_totals == {"1/0001.mxl": 1.0, "1/0002.mxl": 0.5}
    assert collector.stage_stats()[0].stage == "scale_match"
    assert collector.stage_stats()[0].count == 2
    assert collector.stage_stats()[0].min_seconds == 0.25
    assert collector.stage_stats()[0].max_seconds == 0.75


def test_processing_profiler_writes_reports(tmp_path: Path) -> None:
    collector = ProcessingProfiler(output_dir=tmp_path)
    collector.record(stage="scale_match", source_file="1/0001.mxl", seconds=0.25)

    collector.write_reports()

    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "records.csv").exists()
    assert (tmp_path / "stage_stats.csv").exists()
    assert (tmp_path / "source_stats.csv").exists()


def test_processing_profiler_writes_cpu_reports_and_streams_records(tmp_path: Path) -> None:
    with ProcessingProfiler(output_dir=tmp_path, use_torch_profiler_labels=False, retain_records=False) as collector:
        collector.record(stage="scale_match", source_file="1/0001.mxl", seconds=0.25)

    collector.write_reports()

    assert collector.records == ()
    assert (tmp_path / "records.csv").exists()
    assert (tmp_path / "stage_stats.csv").exists()
    assert (tmp_path / "cpu_profile.pstats").exists()
    assert (tmp_path / "cpu_profile_top.txt").exists()
    assert (tmp_path / "cpu_profile_functions.csv").exists()
    assert (tmp_path / "torch_profiler_functions.csv").exists()
