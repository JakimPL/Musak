from pathlib import Path

from musak_model.n_grams.profile.artifacts import figure_artifact_paths


def test_figure_artifact_paths_resolve_under_encoded_run() -> None:
    paths = figure_artifact_paths(Path("processed/PDMX/encoded/abc"))

    assert paths.root_directory == Path("processed/PDMX/encoded/abc/figure")
    assert paths.config_path == Path("processed/PDMX/encoded/abc/figure/config.yml")
    assert paths.profile_path == Path("processed/PDMX/encoded/abc/figure/all/profile.json")
    assert paths.counts_path == Path("processed/PDMX/encoded/abc/figure/all/counts.csv")
    assert paths.base_durations_path == Path("processed/PDMX/encoded/abc/figure/all/base_durations.csv")
    assert paths.by_sample_path == Path("processed/PDMX/encoded/abc/figure/by_sample.jsonl")
