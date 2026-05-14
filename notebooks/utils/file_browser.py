from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUPPORTED_MUSICXML_SUFFIXES = frozenset({".mxl", ".musicxml", ".xml"})


@dataclass(frozen=True)
class FileSelection:
    path: Path | None
    message: str | None
    value_repr: str

    @property
    def has_error(self) -> bool:
        return self.message is not None


def selected_musicxml_file(file_browser: Any) -> FileSelection:
    value_repr = repr(getattr(file_browser, "value", None))

    try:
        selected_path = file_browser.path(0)
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        return FileSelection(
            path=None,
            message=f"Could not read the selected file path: {type(exc).__name__}: {exc}",
            value_repr=value_repr,
        )

    if selected_path is None:
        return FileSelection(
            path=None,
            message="No file is selected. Open folders with the file browser, then select a MusicXML file.",
            value_repr=value_repr,
        )

    path = Path(selected_path)
    if not path.exists():
        return FileSelection(
            path=None,
            message=f"Selected path does not exist: {path}",
            value_repr=value_repr,
        )

    if path.is_dir():
        return FileSelection(
            path=None,
            message=f"Selected path is a directory, not a file: {path}",
            value_repr=value_repr,
        )

    if path.suffix.lower() not in SUPPORTED_MUSICXML_SUFFIXES:
        suffixes = ", ".join(sorted(SUPPORTED_MUSICXML_SUFFIXES))
        return FileSelection(
            path=None,
            message=f"Selected file is not a supported MusicXML file ({suffixes}): {path}",
            value_repr=value_repr,
        )

    return FileSelection(path=path, message=None, value_repr=value_repr)
