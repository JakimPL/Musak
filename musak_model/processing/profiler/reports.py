from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Final

from musak_model.processing.io import JSON_INDENT
from musak_model.processing.profiler.schema import (
    ProcessingProfileRecord,
    ProcessingProfileStageStats,
    ProcessingProfileSummary,
)
from musak_shared.files import write_csv_rows

_JSON_INDENT: Final[int] = JSON_INDENT


def write_summary_report(summary: ProcessingProfileSummary, output_directory: Path) -> None:
    (output_directory / "summary.json").write_text(
        json.dumps(asdict(summary), indent=_JSON_INDENT) + "\n", encoding="utf-8"
    )


def write_records_report(records: list[ProcessingProfileRecord], output_directory: Path) -> None:
    write_csv_rows(
        output_directory / "records.csv",
        columns=tuple(ProcessingProfileRecord.__dataclass_fields__),
        rows=(asdict(record) for record in records),
    )


def write_stage_stats_report(stage_stats: list[ProcessingProfileStageStats], output_directory: Path) -> None:
    write_csv_rows(
        output_directory / "stage_stats.csv",
        columns=tuple(ProcessingProfileStageStats.__dataclass_fields__),
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
