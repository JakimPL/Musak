from pathlib import Path
from typing import Final

MUSICXML_EXTENSIONS: Final[frozenset[str]] = frozenset({".xml", ".mxl", ".musicxml"})


def collect_musicxml_files(source_dir: Path) -> list[Path]:
    patterns = [f"**/*{ext}" for ext in MUSICXML_EXTENSIONS]
    files: set[Path] = set()
    for pattern in patterns:
        files.update(source_dir.rglob(pattern))

    return sorted(files)
