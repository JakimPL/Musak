from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from musak_shared.files import JSON_INDENT, write_csv_rows
from musak_shared.profiling.schema import ProfileRecord, ProfileStageStats, ProfileSummary


def write_summary_report(summary: ProfileSummary, output_directory: Path) -> None:
    (output_directory / "summary.json").write_text(
        json.dumps(asdict(summary), indent=JSON_INDENT) + "\n", encoding="utf-8"
    )


def write_records_report(records: list[ProfileRecord], output_directory: Path) -> None:
    write_csv_rows(
        output_directory / "records.csv",
        columns=tuple(ProfileRecord.__dataclass_fields__),
        rows=(asdict(record) for record in records),
    )


def write_stage_stats_report(stage_stats: list[ProfileStageStats], output_directory: Path) -> None:
    write_csv_rows(
        output_directory / "stage_stats.csv",
        columns=tuple(ProfileStageStats.__dataclass_fields__),
        rows=(asdict(row) for row in stage_stats),
    )


def write_source_stats_report(per_file_totals: dict[str, float], output_directory: Path) -> None:
    write_csv_rows(
        output_directory / "source_stats.csv",
        columns=("source_file", "total_seconds"),
        rows=(
            {"source_file": source_file, "total_seconds": total_seconds}
            for source_file, total_seconds in sorted(per_file_totals.items(), key=lambda item: item[1], reverse=True)
        ),
    )
