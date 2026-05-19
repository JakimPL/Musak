from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from musak_shared.elements import MUSICXML_EXTENSIONS


@dataclass(frozen=True)
class FileSelection:
    path: Path | None
    message: str | None
    value_repr: str

    @property
    def has_error(self) -> bool:
        return self.message is not None


def selected_musicxml_file(file_browser: Any) -> FileSelection:
    return selected_file(
        file_browser,
        supported_suffixes=MUSICXML_EXTENSIONS,
        description="MusicXML",
    )


def selected_file(
    file_browser: Any,
    *,
    supported_suffixes: frozenset[str],
    description: str,
) -> FileSelection:
    value_repr = repr(getattr(file_browser, "value", None))

    try:
        selected_path = file_browser.path(0)
    except (AttributeError, IndexError, TypeError, ValueError) as exception:
        return FileSelection(
            path=None,
            message=f"Could not read the selected file path: {type(exception).__name__}: {exception}",
            value_repr=value_repr,
        )

    if selected_path is None:
        return FileSelection(
            path=None,
            message=f"No file is selected. Open folders with the file browser, then select a {description} file.",
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

    if path.suffix.lower() not in supported_suffixes:
        suffixes = ", ".join(sorted(supported_suffixes))
        return FileSelection(
            path=None,
            message=f"Selected file is not a supported {description} file ({suffixes}): {path}",
            value_repr=value_repr,
        )

    return FileSelection(path=path, message=None, value_repr=value_repr)
