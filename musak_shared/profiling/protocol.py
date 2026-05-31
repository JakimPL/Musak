from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self


class ProfilerProtocol(Protocol):
    @property
    def enabled(self) -> bool: ...

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    def measure(self, stage: str, *, source_file: Path | None = None) -> AbstractContextManager[None]: ...

    def step(self) -> None: ...

    def write_reports(self) -> None: ...


class ProfilerBackend(Protocol):
    def session(self) -> AbstractContextManager[None]: ...

    def span(self, stage: str) -> AbstractContextManager[None]: ...

    def before_measure(self) -> None: ...

    def after_measure(self) -> None: ...

    def write_reports(self, output_directory: Path) -> None: ...
