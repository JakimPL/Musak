import csv
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from musak_model.data.schema import ParsedScore, Segment
from musak_model.evaluation import diagnose_segment
from musak_model.processing.ids import segment_id
from musak_model.tokens.duration import DurationVocabulary
from musak_shared.ratios import format_ratio

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
    PARSE_DIAGNOSTICS = "parse_diagnostics"
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
    TOKEN_COUNT = "token_count"
    ELIGIBLE_FOR_TRAINING = "eligible_for_training"
    INELIGIBILITY_REASONS = "ineligibility_reasons"
    KEY_ROOT = "key_root"
    SCALE_TYPE = "scale_type"
    TIME_SIGNATURE = "time_signature"
    DIFFICULTY_LEVEL = "difficulty_level"
    RIGHT_SILENCE_FRACTION = "right_silence_fraction"
    LEFT_SILENCE_FRACTION = "left_silence_fraction"
    BOTH_HANDS_SILENCE_FRACTION = "both_hands_silence_fraction"
    BOTH_HANDS_ACTIVE_FRACTION = "both_hands_active_fraction"
    RIGHT_ONLY_ACTIVE_FRACTION = "right_only_active_fraction"
    LEFT_ONLY_ACTIVE_FRACTION = "left_only_active_fraction"
    LONGEST_RIGHT_SILENCE_BEATS = "longest_right_silence_beats"
    LONGEST_LEFT_SILENCE_BEATS = "longest_left_silence_beats"
    LONGEST_BOTH_HANDS_SILENCE_BEATS = "longest_both_hands_silence_beats"
    RIGHT_NOTE_ONSETS_PER_BAR = "right_note_onsets_per_bar"
    LEFT_NOTE_ONSETS_PER_BAR = "left_note_onsets_per_bar"
    SILENT_BAR_COUNT = "silent_bar_count"
    SILENT_BAR_FRACTION = "silent_bar_fraction"
    SILENT_EDGE_BAR_COUNT = "silent_edge_bar_count"
    HAND_ACTIVITY_BALANCE = "hand_activity_balance"
    EMPTY_SCORE = "empty_score"
    ONE_HAND_ONLY = "one_hand_only"
    NOTE_TOKEN_FRACTION = "note_token_fraction"
    REST_TOKEN_FRACTION = "rest_token_fraction"
    HOLD_TOKEN_FRACTION = "hold_token_fraction"


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
    ParsedManifestField.PARSE_DIAGNOSTICS,
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
    EncodedManifestField.TOKEN_COUNT,
    EncodedManifestField.ELIGIBLE_FOR_TRAINING,
    EncodedManifestField.INELIGIBILITY_REASONS,
    EncodedManifestField.KEY_ROOT,
    EncodedManifestField.SCALE_TYPE,
    EncodedManifestField.TIME_SIGNATURE,
    EncodedManifestField.DIFFICULTY_LEVEL,
    EncodedManifestField.RIGHT_SILENCE_FRACTION,
    EncodedManifestField.LEFT_SILENCE_FRACTION,
    EncodedManifestField.BOTH_HANDS_SILENCE_FRACTION,
    EncodedManifestField.BOTH_HANDS_ACTIVE_FRACTION,
    EncodedManifestField.RIGHT_ONLY_ACTIVE_FRACTION,
    EncodedManifestField.LEFT_ONLY_ACTIVE_FRACTION,
    EncodedManifestField.LONGEST_RIGHT_SILENCE_BEATS,
    EncodedManifestField.LONGEST_LEFT_SILENCE_BEATS,
    EncodedManifestField.LONGEST_BOTH_HANDS_SILENCE_BEATS,
    EncodedManifestField.RIGHT_NOTE_ONSETS_PER_BAR,
    EncodedManifestField.LEFT_NOTE_ONSETS_PER_BAR,
    EncodedManifestField.SILENT_BAR_COUNT,
    EncodedManifestField.SILENT_BAR_FRACTION,
    EncodedManifestField.SILENT_EDGE_BAR_COUNT,
    EncodedManifestField.HAND_ACTIVITY_BALANCE,
    EncodedManifestField.EMPTY_SCORE,
    EncodedManifestField.ONE_HAND_ONLY,
    EncodedManifestField.NOTE_TOKEN_FRACTION,
    EncodedManifestField.REST_TOKEN_FRACTION,
    EncodedManifestField.HOLD_TOKEN_FRACTION,
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
    parse_diagnostics: str,
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
        ParsedManifestField.PARSE_DIAGNOSTICS: parse_diagnostics,
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
    parse_diagnostics: str,
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
        ParsedManifestField.PARSE_DIAGNOSTICS: parse_diagnostics,
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
    duration_vocabulary: DurationVocabulary,
    encoded_sample: "EncodedExercise | None",
    encoded_shard: Path,
    encoded_line: int | None,
) -> dict[str, Any]:
    diagnostics = diagnose_segment(segment, duration_vocabulary=duration_vocabulary)
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
        EncodedManifestField.TOKEN_COUNT: len(segment.tokens),
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
        EncodedManifestField.RIGHT_SILENCE_FRACTION: diagnostics.right_silence_fraction,
        EncodedManifestField.LEFT_SILENCE_FRACTION: diagnostics.left_silence_fraction,
        EncodedManifestField.BOTH_HANDS_SILENCE_FRACTION: diagnostics.both_hands_silence_fraction,
        EncodedManifestField.BOTH_HANDS_ACTIVE_FRACTION: diagnostics.both_hands_active_fraction,
        EncodedManifestField.RIGHT_ONLY_ACTIVE_FRACTION: diagnostics.right_only_active_fraction,
        EncodedManifestField.LEFT_ONLY_ACTIVE_FRACTION: diagnostics.left_only_active_fraction,
        EncodedManifestField.LONGEST_RIGHT_SILENCE_BEATS: diagnostics.longest_right_silence_beats,
        EncodedManifestField.LONGEST_LEFT_SILENCE_BEATS: diagnostics.longest_left_silence_beats,
        EncodedManifestField.LONGEST_BOTH_HANDS_SILENCE_BEATS: diagnostics.longest_both_hands_silence_beats,
        EncodedManifestField.RIGHT_NOTE_ONSETS_PER_BAR: diagnostics.right_note_onsets_per_bar,
        EncodedManifestField.LEFT_NOTE_ONSETS_PER_BAR: diagnostics.left_note_onsets_per_bar,
        EncodedManifestField.SILENT_BAR_COUNT: diagnostics.silent_bar_count,
        EncodedManifestField.SILENT_BAR_FRACTION: diagnostics.silent_bar_fraction,
        EncodedManifestField.SILENT_EDGE_BAR_COUNT: diagnostics.silent_edge_bar_count,
        EncodedManifestField.HAND_ACTIVITY_BALANCE: diagnostics.hand_activity_balance,
        EncodedManifestField.EMPTY_SCORE: diagnostics.empty_score,
        EncodedManifestField.ONE_HAND_ONLY: diagnostics.one_hand_only,
        EncodedManifestField.NOTE_TOKEN_FRACTION: diagnostics.note_token_fraction,
        EncodedManifestField.REST_TOKEN_FRACTION: diagnostics.rest_token_fraction,
        EncodedManifestField.HOLD_TOKEN_FRACTION: diagnostics.hold_token_fraction,
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
