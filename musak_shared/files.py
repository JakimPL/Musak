import shutil
from pathlib import Path
from typing import Any

import yaml

from musak_shared.elements import MUSICXML_EXTENSIONS


def collect_musicxml_files(source_directory: Path) -> list[Path]:
    patterns = [f"**/*{ext}" for ext in MUSICXML_EXTENSIONS]
    files: set[Path] = set()
    for pattern in patterns:
        files.update(source_directory.rglob(pattern))

    return sorted(files)


def load_yaml_config(path: Path) -> dict[str, Any]:
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"expected mapping in config file: {path}")

    return parsed


def write_yaml_config(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")
    temp_path.replace(path)


def line_count(path: Path) -> int:
    if not path.exists():
        return 0

    with path.open("r", encoding="utf-8") as file:
        return sum(1 for _ in file)


def append_text_lines_from_index(
    source_path: Path,
    destination_path: Path,
    *,
    start_line_index: int,
) -> dict[int, int]:
    line_mapping: dict[int, int] = {}
    if not source_path.exists():
        return line_mapping

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("r", encoding="utf-8") as source_file:
        with destination_path.open("a", encoding="utf-8") as destination_file:
            for local_line, line in enumerate(source_file):
                line_mapping[local_line] = start_line_index + local_line
                destination_file.write(line)

    return line_mapping


def truncate_text_lines(path: Path, line_count_value: int) -> None:
    if not path.exists():
        return

    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("".join(f"{line}\n" for line in lines[:line_count_value]), encoding="utf-8")


def remove_directory_tree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def move_path(source_path: Path, destination_path: Path) -> None:
    shutil.move(str(source_path), str(destination_path))


def remove_empty_parents(path: Path, *, stop_at: Path) -> None:
    current_path = path
    while current_path != stop_at and current_path.exists():
        try:
            current_path.rmdir()
        except OSError:
            return
        current_path = current_path.parent
