from __future__ import annotations

import cProfile
import pstats
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from typing import Final

from musak_shared.files import write_csv_rows

_CPU_PROFILE_STATS_NAME: Final[str] = "cpu_profile.pstats"
_CPU_PROFILE_TABLE_NAME: Final[str] = "cpu_profile_top.txt"
_CPU_PROFILE_FUNCTIONS_NAME: Final[str] = "cpu_profile_functions.csv"
_CPU_PROFILE_ROW_LIMIT: Final[int] = 200


class CProfileBackend:
    def __init__(self) -> None:
        self._profile: cProfile.Profile | None = None

    @contextmanager
    def session(self) -> Iterator[None]:
        self._profile = cProfile.Profile()
        self._profile.enable()
        try:
            yield
        finally:
            self._profile.disable()

    def span(self, stage: str) -> AbstractContextManager[None]:
        _ = stage
        return nullcontext()

    def before_measure(self) -> None:
        return None

    def after_measure(self) -> None:
        return None

    def write_reports(self, output_directory: Path) -> None:
        if self._profile is not None:
            _write_cpu_reports(self._profile, output_directory)


def _write_cpu_reports(profile: cProfile.Profile, output_directory: Path) -> None:
    stats_path = output_directory / _CPU_PROFILE_STATS_NAME
    table_path = output_directory / _CPU_PROFILE_TABLE_NAME
    functions_path = output_directory / _CPU_PROFILE_FUNCTIONS_NAME

    profile.dump_stats(str(stats_path))
    with table_path.open("w", encoding="utf-8") as file:
        stats = pstats.Stats(profile, stream=file)
        stats.sort_stats(pstats.SortKey.CUMULATIVE)
        stats.print_stats(_CPU_PROFILE_ROW_LIMIT)

    stats = pstats.Stats(profile)
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    _write_function_stats(stats.get_stats_profile(), functions_path)


def _write_function_stats(stats: pstats.StatsProfile, path: Path) -> None:
    rows = sorted(
        (
            _function_row(function_name, function_profile)
            for function_name, function_profile in stats.func_profiles.items()
        ),
        key=lambda row: row["cumulative_seconds"],
        reverse=True,
    )
    write_csv_rows(
        path,
        columns=(
            "function",
            "file",
            "line",
            "calls",
            "total_seconds",
            "cumulative_seconds",
            "per_call_seconds",
        ),
        rows=rows,
    )


def _function_row(
    function_name: str,
    function_profile: pstats.FunctionProfile,
) -> dict[str, str | int | float]:
    return {
        "function": function_name,
        "file": function_profile.file_name,
        "line": function_profile.line_number,
        "calls": function_profile.ncalls,
        "total_seconds": function_profile.tottime,
        "cumulative_seconds": function_profile.cumtime,
        "per_call_seconds": function_profile.percall_tottime,
    }
