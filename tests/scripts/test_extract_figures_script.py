from pathlib import Path

import pytest

from musak_model.analysis.n_grams.profile.extraction import copy_analysis_config
from scripts.extract_figures import (
    dataset_name_for_analysis,
    default_output_path,
    encoded_run_dirs,
    resolve_encoded_dir,
)


def test_resolve_encoded_dir_uses_single_processed_run(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "PDMX"
    encoded_dir = tmp_path / "processed" / "PDMX" / "encoded" / "abc"
    _write_encoded_run(encoded_dir)

    resolved = resolve_encoded_dir(
        data_dir=data_dir,
        processed_root=tmp_path / "processed",
        encoded_dir=None,
    )

    assert resolved == encoded_dir


def test_resolve_encoded_dir_requires_override_for_multiple_runs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "PDMX"
    _write_encoded_run(tmp_path / "processed" / "PDMX" / "encoded" / "abc")
    _write_encoded_run(tmp_path / "processed" / "PDMX" / "encoded" / "def")

    with pytest.raises(ValueError, match="Multiple encoded runs"):
        resolve_encoded_dir(
            data_dir=data_dir,
            processed_root=tmp_path / "processed",
            encoded_dir=None,
        )


def test_resolve_encoded_dir_prefers_explicit_encoded_dir(tmp_path: Path) -> None:
    encoded_dir = tmp_path / "custom"

    resolved = resolve_encoded_dir(
        data_dir=None,
        processed_root=tmp_path / "processed",
        encoded_dir=encoded_dir,
    )

    assert resolved == encoded_dir


def test_encoded_run_dirs_requires_jsonl_and_tokenizer_snapshot(tmp_path: Path) -> None:
    valid = tmp_path / "encoded" / "valid"
    _write_encoded_run(valid)
    invalid = tmp_path / "encoded" / "invalid"
    invalid.mkdir(parents=True)
    (invalid / "data-00000.jsonl").write_text("", encoding="utf-8")

    assert encoded_run_dirs(tmp_path / "encoded") == [valid]


def test_dataset_name_prefers_data_dir() -> None:
    name = dataset_name_for_analysis(
        data_dir=Path("data/Exercises"),
        encoded_dir=Path("processed/PDMX/encoded/abc"),
    )

    assert name == "Exercises"


def test_default_output_path_uses_analysis_directory() -> None:
    path = default_output_path(
        data_dir=Path("data/PDMX"),
        encoded_dir=Path("processed/PDMX/encoded/abc"),
    )

    assert path == Path("processed/PDMX/encoded/abc/figure/all/counts.csv")


def test_copy_analysis_config_writes_canonical_config(tmp_path: Path) -> None:
    source = tmp_path / "n_grams.yml"
    target = tmp_path / "encoded" / "figure" / "config.yml"
    source.write_text("min_n: 1\n", encoding="utf-8")

    copy_analysis_config(source, target)

    assert target.read_text(encoding="utf-8") == "min_n: 1\n"


def _write_encoded_run(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "data-00000.jsonl").write_text("", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
