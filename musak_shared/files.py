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
