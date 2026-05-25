from __future__ import annotations

import csv
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

_JSON_INDENT: Final[int] = JSON_INDENT


def write_summary_report(summary: ProcessingProfileSummary, output_directory: Path) -> None:
    (output_directory / "summary.json").write_text(
        json.dumps(asdict(summary), indent=_JSON_INDENT) + "\n", encoding="utf-8"
    )


def write_records_report(records: list[ProcessingProfileRecord], output_directory: Path) -> None:
    with (output_directory / "records.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(ProcessingProfileRecord.__dataclass_fields__))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_stage_stats_report(stage_stats: list[ProcessingProfileStageStats], output_directory: Path) -> None:
    with (output_directory / "stage_stats.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(ProcessingProfileStageStats.__dataclass_fields__))
        writer.writeheader()
        for row in stage_stats:
            writer.writerow(asdict(row))


def write_source_stats_report(per_file_totals: dict[str, float], output_directory: Path) -> None:
    with (output_directory / "source_stats.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["source_file", "total_seconds"])
        writer.writeheader()
        for source_file, total_seconds in sorted(per_file_totals.items(), key=lambda item: item[1], reverse=True):
            writer.writerow({"source_file": source_file, "total_seconds": total_seconds})
