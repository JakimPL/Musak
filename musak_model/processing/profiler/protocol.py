from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol


class ProcessingProfilerProtocol(Protocol):
    @property
    def enabled(self) -> bool: ...

    def measure(self, stage: str, *, source_file: Path | None = None) -> AbstractContextManager[None]: ...

    def step(self) -> None: ...

    def write_reports(self) -> None: ...
