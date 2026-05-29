from pathlib import Path

import polars as pl
import pytest

from musak_shared.tables import read_table, write_table


def _frame() -> pl.DataFrame:
    return pl.DataFrame({"scale_type": ["major", "minor"], "n": [2, 3], "count": [5, 1]})


def test_parquet_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "counts.parquet"

    write_table(_frame(), path)

    assert path.is_file()
    assert read_table(path).equals(_frame())


def test_csv_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "counts.csv"

    write_table(_frame(), path)

    assert read_table(path).equals(_frame())


def test_write_table_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "counts.parquet"

    write_table(_frame(), path)

    assert path.is_file()


def test_write_table_leaves_no_temporary_file(tmp_path: Path) -> None:
    path = tmp_path / "counts.parquet"

    write_table(_frame(), path)

    assert list(tmp_path.iterdir()) == [path]


def test_write_table_rejects_unsupported_suffix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported table suffix"):
        write_table(_frame(), tmp_path / "counts.json")


def test_read_table_rejects_unsupported_suffix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported table suffix"):
        read_table(tmp_path / "counts.json")
