from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from types import TracebackType
from typing import Self


class NullProfiler:
    @property
    def enabled(self) -> bool:
        return False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def measure(self, stage: str, *, source_file: Path | None = None) -> AbstractContextManager[None]:
        _ = stage, source_file
        return nullcontext()

    def step(self) -> None:
        return None

    def write_reports(self) -> None:
        return None


NULL_PROFILER = NullProfiler()
