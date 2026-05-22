from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from types import TracebackType
from typing import Self


class NullProcessingProfiler:
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
        return nullcontext()

    def step(self) -> None:
        return None

    def write_reports(self) -> None:
        return None


NULL_PROCESSING_PROFILER = NullProcessingProfiler()
