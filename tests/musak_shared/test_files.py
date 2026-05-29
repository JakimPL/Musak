import csv
from pathlib import Path

from musak_shared.files import (
    append_text_lines_from_index,
    line_count,
    move_path,
    remove_directory_tree,
    remove_empty_parents,
    truncate_text_lines,
    write_csv_rows,
)


def test_write_csv_rows_writes_header_and_rows(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "rows.csv"

    write_csv_rows(
        path,
        columns=("stage", "seconds"),
        rows=[{"stage": "parse", "seconds": 1}, {"stage": "tokenize", "seconds": 2}],
    )

    with path.open("r", encoding="utf-8", newline="") as file:
        records = list(csv.DictReader(file))
    assert records == [{"stage": "parse", "seconds": "1"}, {"stage": "tokenize", "seconds": "2"}]


def test_line_count_returns_zero_for_missing_file(tmp_path: Path) -> None:
    assert line_count(tmp_path / "missing.txt") == 0


def test_line_count_counts_text_lines(tmp_path: Path) -> None:
    path = tmp_path / "lines.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")

    assert line_count(path) == 3


def test_append_text_lines_from_index_appends_and_maps_line_numbers(tmp_path: Path) -> None:
    source_path = tmp_path / "source.txt"
    destination_path = tmp_path / "nested" / "destination.txt"
    source_path.write_text("a\nb\n", encoding="utf-8")

    line_mapping = append_text_lines_from_index(source_path, destination_path, start_line_index=5)

    assert line_mapping == {0: 5, 1: 6}
    assert destination_path.read_text(encoding="utf-8") == "a\nb\n"


def test_append_text_lines_from_index_returns_empty_mapping_for_missing_source(tmp_path: Path) -> None:
    line_mapping = append_text_lines_from_index(
        tmp_path / "missing.txt",
        tmp_path / "destination.txt",
        start_line_index=3,
    )

    assert line_mapping == {}


def test_truncate_text_lines_keeps_prefix_lines(tmp_path: Path) -> None:
    path = tmp_path / "lines.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")

    truncate_text_lines(path, 2)

    assert path.read_text(encoding="utf-8") == "a\nb\n"


def test_remove_directory_tree_removes_existing_directory(tmp_path: Path) -> None:
    path = tmp_path / "directory"
    path.mkdir()
    (path / "file.txt").write_text("content", encoding="utf-8")

    remove_directory_tree(path)

    assert not path.exists()


def test_move_path_moves_file(tmp_path: Path) -> None:
    source_path = tmp_path / "source.txt"
    destination_path = tmp_path / "destination.txt"
    source_path.write_text("content", encoding="utf-8")

    move_path(source_path, destination_path)

    assert not source_path.exists()
    assert destination_path.read_text(encoding="utf-8") == "content"


def test_remove_empty_parents_removes_empty_parent_chain(tmp_path: Path) -> None:
    nested_path = tmp_path / "a" / "b" / "c"
    nested_path.mkdir(parents=True)

    remove_empty_parents(nested_path, stop_at=tmp_path)

    assert not (tmp_path / "a").exists()
    assert tmp_path.exists()
