import csv
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from musak_model.common.ratios import format_ratio
from musak_model.data.schema import ParsedScore, Segment
from musak_model.processing.ids import segment_id

if TYPE_CHECKING:
    from musak_model.training.ingestion.schema import EncodedExercise


class ParsedManifestField(StrEnum):
    SOURCE_ID = "source_id"
    SOURCE_PATH = "source_path"
    SOURCE_FILENAME = "source_filename"
    TITLE = "title"
    PARSED_PATH = "parsed_path"
    STATUS = "status"
    ERROR_TYPE = "error_type"
    ERROR_MESSAGE = "error_message"
    RIGHT_HAND_BARS = "right_hand_bars"
    LEFT_HAND_BARS = "left_hand_bars"
    KEY_ROOT = "key_root"
    KEY_FIFTHS = "key_fifths"
    SCALE_TYPE = "scale_type"
    TIME_SIGNATURE = "time_signature"


class EncodedManifestField(StrEnum):
    SEGMENT_ID = "segment_id"
    SOURCE_ID = "source_id"
    SOURCE_PATH = "source_path"
    PARSED_PATH = "parsed_path"
    ENCODED_SHARD = "encoded_shard"
    ENCODED_LINE = "encoded_line"
    WINDOW_START_BAR = "window_start_bar"
    BAR_COUNT = "bar_count"
    ELIGIBLE_FOR_TRAINING = "eligible_for_training"
    INELIGIBILITY_REASONS = "ineligibility_reasons"
    KEY_ROOT = "key_root"
    SCALE_TYPE = "scale_type"
    TIME_SIGNATURE = "time_signature"
    DIFFICULTY_LEVEL = "difficulty_level"


class ParsedManifestStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"


PARSED_MANIFEST_FIELDS: Final[tuple[ParsedManifestField, ...]] = (
    ParsedManifestField.SOURCE_ID,
    ParsedManifestField.SOURCE_PATH,
    ParsedManifestField.SOURCE_FILENAME,
    ParsedManifestField.TITLE,
    ParsedManifestField.PARSED_PATH,
    ParsedManifestField.STATUS,
    ParsedManifestField.ERROR_TYPE,
    ParsedManifestField.ERROR_MESSAGE,
    ParsedManifestField.RIGHT_HAND_BARS,
    ParsedManifestField.LEFT_HAND_BARS,
    ParsedManifestField.KEY_ROOT,
    ParsedManifestField.KEY_FIFTHS,
    ParsedManifestField.SCALE_TYPE,
    ParsedManifestField.TIME_SIGNATURE,
)

ENCODED_MANIFEST_FIELDS: Final[tuple[EncodedManifestField, ...]] = (
    EncodedManifestField.SEGMENT_ID,
    EncodedManifestField.SOURCE_ID,
    EncodedManifestField.SOURCE_PATH,
    EncodedManifestField.PARSED_PATH,
    EncodedManifestField.ENCODED_SHARD,
    EncodedManifestField.ENCODED_LINE,
    EncodedManifestField.WINDOW_START_BAR,
    EncodedManifestField.BAR_COUNT,
    EncodedManifestField.ELIGIBLE_FOR_TRAINING,
    EncodedManifestField.INELIGIBILITY_REASONS,
    EncodedManifestField.KEY_ROOT,
    EncodedManifestField.SCALE_TYPE,
    EncodedManifestField.TIME_SIGNATURE,
    EncodedManifestField.DIFFICULTY_LEVEL,
)


def write_parsed_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    _write_manifest_csv(rows, path, fieldnames=_field_values(PARSED_MANIFEST_FIELDS))


def write_encoded_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    _write_manifest_csv(rows, path, fieldnames=_field_values(ENCODED_MANIFEST_FIELDS))


def read_parsed_manifest(path: Path) -> list[dict[str, str]]:
    return _read_manifest_csv(path)


def read_encoded_manifest(path: Path) -> list[dict[str, str]]:
    return _read_manifest_csv(path)


def parsed_success_row(
    *,
    source_id_value: str,
    source_path: Path,
    dataset_root: Path,
    title: str,
    parsed_path: Path,
    processed_root: Path,
    score: ParsedScore,
) -> dict[str, Any]:
    return {
        ParsedManifestField.SOURCE_ID: source_id_value,
        ParsedManifestField.SOURCE_PATH: _relative_text(source_path, dataset_root),
        ParsedManifestField.SOURCE_FILENAME: source_path.name,
        ParsedManifestField.TITLE: title,
        ParsedManifestField.PARSED_PATH: _relative_text(parsed_path, processed_root),
        ParsedManifestField.STATUS: ParsedManifestStatus.SUCCESS.value,
        ParsedManifestField.ERROR_TYPE: "",
        ParsedManifestField.ERROR_MESSAGE: "",
        ParsedManifestField.RIGHT_HAND_BARS: len(score.right_hand_bars),
        ParsedManifestField.LEFT_HAND_BARS: len(score.left_hand_bars),
        ParsedManifestField.KEY_ROOT: score.key_root,
        ParsedManifestField.KEY_FIFTHS: score.key_fifths,
        ParsedManifestField.SCALE_TYPE: score.scale_type.value,
        ParsedManifestField.TIME_SIGNATURE: format_ratio((score.time_numerator, score.time_denominator)),
    }


def parsed_error_row(
    *,
    source_id_value: str,
    source_path: Path,
    dataset_root: Path,
    title: str,
    exception: Exception,
) -> dict[str, Any]:
    return {
        ParsedManifestField.SOURCE_ID: source_id_value,
        ParsedManifestField.SOURCE_PATH: _relative_text(source_path, dataset_root),
        ParsedManifestField.SOURCE_FILENAME: source_path.name,
        ParsedManifestField.TITLE: title,
        ParsedManifestField.PARSED_PATH: "",
        ParsedManifestField.STATUS: ParsedManifestStatus.ERROR.value,
        ParsedManifestField.ERROR_TYPE: type(exception).__name__,
        ParsedManifestField.ERROR_MESSAGE: str(exception),
        ParsedManifestField.RIGHT_HAND_BARS: "",
        ParsedManifestField.LEFT_HAND_BARS: "",
        ParsedManifestField.KEY_ROOT: "",
        ParsedManifestField.KEY_FIFTHS: "",
        ParsedManifestField.SCALE_TYPE: "",
        ParsedManifestField.TIME_SIGNATURE: "",
    }


def encoded_row(
    *,
    source_id_value: str,
    source_path: Path,
    dataset_root: Path,
    parsed_path: Path,
    processed_root: Path,
    segment: Segment,
    encoded_sample: "EncodedExercise | None",
    encoded_shard: Path,
    encoded_line: int | None,
) -> dict[str, Any]:
    return {
        EncodedManifestField.SEGMENT_ID: segment_id(
            source_id_value,
            window_start_bar=segment.metadata.window_start_bar,
            bar_count=segment.metadata.bar_count,
        ),
        EncodedManifestField.SOURCE_ID: source_id_value,
        EncodedManifestField.SOURCE_PATH: _relative_text(source_path, dataset_root),
        EncodedManifestField.PARSED_PATH: _relative_text(parsed_path, processed_root),
        EncodedManifestField.ENCODED_SHARD: (
            _relative_text(encoded_shard, processed_root) if encoded_sample is not None else ""
        ),
        EncodedManifestField.ENCODED_LINE: encoded_line if encoded_line is not None else "",
        EncodedManifestField.WINDOW_START_BAR: segment.metadata.window_start_bar,
        EncodedManifestField.BAR_COUNT: segment.metadata.bar_count,
        EncodedManifestField.ELIGIBLE_FOR_TRAINING: segment.metadata.eligible_for_training,
        EncodedManifestField.INELIGIBILITY_REASONS: "|".join(
            sorted(reason.value for reason in segment.metadata.ineligibility_reasons)
        ),
        EncodedManifestField.KEY_ROOT: segment.metadata.key_root,
        EncodedManifestField.SCALE_TYPE: segment.metadata.scale_type.value,
        EncodedManifestField.TIME_SIGNATURE: format_ratio(
            (segment.metadata.time_numerator, segment.metadata.time_denominator)
        ),
        EncodedManifestField.DIFFICULTY_LEVEL: (
            segment.metadata.difficulty_level if segment.metadata.difficulty_level is not None else ""
        ),
    }


def _field_values(fields: tuple[StrEnum, ...]) -> tuple[str, ...]:
    return tuple(field.value for field in fields)


def _write_manifest_csv(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    fieldnames: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_manifest_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _relative_text(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()
