from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import pandas as pd

PROFILE_FILES: Final[tuple[str, ...]] = (
    "summary.json",
    "stage_stats.csv",
    "source_stats.csv",
    "cpu_profile_functions.csv",
    "torch_profiler_functions.csv",
    "cpu_profile_top.txt",
    "torch_profiler_table.txt",
    "records.csv",
)
DEFAULT_SORT_COLUMN: Final[str] = "total_seconds"


def existing_directory(
    path: Path,
    *,
    fallback: Path,
) -> Path:
    current = path
    while not current.exists() or not current.is_dir():
        if current == current.parent:
            return fallback

        current = current.parent

    return current


def selected_profile_root(
    browser: Any,
    *,
    default: Path,
) -> Path:
    try:
        selected_path = browser.path(0)
    except (AttributeError, IndexError, TypeError, ValueError):
        return default

    if selected_path is None:
        return default

    return Path(selected_path).expanduser()


def read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"profile JSON must contain an object: {path}")

    return {str(key): value for key, value in payload.items()}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def file_status_rows(profile_dir: Path) -> list[dict[str, str | bool]]:
    return [
        {
            "artifact": file_name,
            "path": str(profile_dir / file_name),
            "exists": (profile_dir / file_name).exists(),
        }
        for file_name in PROFILE_FILES
    ]


def profile_artifact_count(profile_dir: Path) -> int:
    return sum(1 for file_name in PROFILE_FILES if (profile_dir / file_name).exists())


def profile_mode_paths(profile_root: Path) -> dict[str, Path]:
    candidates: dict[str, Path] = {}
    if profile_artifact_count(profile_root) > 0:
        candidates["selected directory"] = profile_root
    if profile_root.exists():
        for path in _ordered_profile_mode_dirs(profile_root):
            if profile_artifact_count(path) > 0:
                candidates[path.name] = path

    return candidates


def sorted_frame(frame: pd.DataFrame, *, sort_column: str | None = None) -> pd.DataFrame:
    selected_sort_column = _sort_column(frame, fallback=sort_column)
    if frame.empty or selected_sort_column is None:
        return frame

    return frame.sort_values(selected_sort_column, ascending=False)


def chart_frame(frame: pd.DataFrame, *, row_count: int) -> pd.DataFrame:
    if frame.empty:
        return frame

    return frame.head(row_count)


def has_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> bool:
    return not frame.empty and all(column in frame.columns for column in columns)


def percentage_frame(
    frame: pd.DataFrame,
    *,
    value_column: str,
    total: float,
    output_column: str,
) -> pd.DataFrame:
    if frame.empty or value_column not in frame.columns or total <= 0:
        return frame

    result = frame.copy()
    result[output_column] = result[value_column] / total
    return result


def metric_rows(
    summary: dict[str, object],
    stage_stats: pd.DataFrame,
    source_stats: pd.DataFrame,
    cpu_functions: pd.DataFrame,
    torch_functions: pd.DataFrame,
    records: pd.DataFrame,
) -> list[dict[str, str | float]]:
    return [
        {"metric": "profiled_seconds", "value": _float_value(summary.get("total_seconds", 0.0))},
        {"metric": "measured_stage_count", "value": float(len(stage_stats))},
        {"metric": "measured_source_count", "value": float(len(source_stats))},
        {"metric": "cpu_function_count", "value": float(len(cpu_functions))},
        {"metric": "torch_operation_count", "value": float(len(torch_functions))},
        {"metric": "raw_record_count_loaded", "value": float(len(records))},
    ]


def _ordered_profile_mode_dirs(profile_root: Path) -> list[Path]:
    stage_names = ("parse", "tokenize", "process")
    stage_paths = [profile_root / stage_name for stage_name in stage_names]
    other_paths = [path for path in sorted(profile_root.iterdir()) if path.is_dir() and path.name not in stage_names]
    return [*stage_paths, *other_paths]


def _float_value(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)

    return 0.0


def _sort_column(frame: pd.DataFrame, *, fallback: str | None) -> str | None:
    if DEFAULT_SORT_COLUMN in frame.columns:
        return DEFAULT_SORT_COLUMN

    if fallback is not None and fallback in frame.columns:
        return fallback

    return None
