from __future__ import annotations

import warnings
from contextlib import redirect_stderr
from io import StringIO
from types import TracebackType
from typing import Any, Final

_DIAGNOSTIC_SEPARATOR: Final[str] = "\n"


class ParseDiagnosticsCapture:
    def __init__(self) -> None:
        self._stderr = StringIO()
        self._warning_records: list[warnings.WarningMessage] = []
        self._warnings_context: Any | None = None
        self._stderr_context: redirect_stderr[StringIO] | None = None

    def __enter__(self) -> ParseDiagnosticsCapture:
        warnings_context = warnings.catch_warnings(record=True)
        self._warnings_context = warnings_context
        self._warning_records = warnings_context.__enter__()
        warnings.simplefilter("always")
        self._stderr_context = redirect_stderr(self._stderr)
        self._stderr_context.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._stderr_context is not None:
            self._stderr_context.__exit__(exc_type, exc_value, traceback)

        if self._warnings_context is not None:
            self._warnings_context.__exit__(exc_type, exc_value, traceback)

    def text(self) -> str:
        messages = [self._format_warning(record) for record in self._warning_records]
        stderr_text = self._stderr.getvalue().strip()
        if stderr_text:
            messages.append(stderr_text)

        return _DIAGNOSTIC_SEPARATOR.join(message for message in messages if message)

    @staticmethod
    def _format_warning(record: warnings.WarningMessage) -> str:
        return f"{record.category.__name__}: {record.message}"
