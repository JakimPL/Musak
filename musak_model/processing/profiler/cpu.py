from __future__ import annotations

import cProfile
import csv
import pstats
from pathlib import Path
from typing import Final

_CPU_PROFILE_STATS_NAME: Final[str] = "cpu_profile.pstats"
_CPU_PROFILE_TABLE_NAME: Final[str] = "cpu_profile_top.txt"
_CPU_PROFILE_FUNCTIONS_NAME: Final[str] = "cpu_profile_functions.csv"
_CPU_PROFILE_ROW_LIMIT: Final[int] = 200


def create_cpu_profiler() -> cProfile.Profile:
    return cProfile.Profile()


def write_cpu_reports(profile: cProfile.Profile, output_dir: Path) -> None:
    stats_path = output_dir / _CPU_PROFILE_STATS_NAME
    table_path = output_dir / _CPU_PROFILE_TABLE_NAME
    functions_path = output_dir / _CPU_PROFILE_FUNCTIONS_NAME

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
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "function",
                "file",
                "line",
                "calls",
                "total_seconds",
                "cumulative_seconds",
                "per_call_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


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
