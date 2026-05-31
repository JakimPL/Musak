import csv
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from musak_model.data.config import SegmentationMode
from musak_model.data.schema import ParsedScore, Segment
from musak_model.evaluation.diagnostics import SegmentDiagnostics
from musak_model.processing.ids import segment_id
from musak_shared.files import write_csv_rows
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
    DECLARED_KEY_FIFTHS = "declared_key_fifths"
    TIME_SIGNATURE = "time_signature"


class EncodedManifestField(StrEnum):
    SEGMENT_ID = "segment_id"
    SOURCE_ID = "source_id"
    SOURCE_PATH = "source_path"
    PARSED_PATH = "parsed_path"
    ENCODED_SHARD = "encoded_shard"
    ENCODED_LINE = "encoded_line"
    SEGMENTATION_MODE = "segmentation_mode"
    WINDOW_START_BAR = "window_start_bar"
    BAR_COUNT = "bar_count"
    TOKEN_COUNT = "token_count"
    ELIGIBLE_FOR_TRAINING = "eligible_for_training"
    INELIGIBILITY_REASONS = "ineligibility_reasons"
    SCALE_ROOT = "scale_root"
    SCALE_TYPE = "scale_type"
    DECLARED_KEY_FIFTHS = "declared_key_fifths"
    SPELLING_KEY_FIFTHS = "spelling_key_fifths"
    SPELLING_CONTEXT_SOURCE = "spelling_context_source"
    SCALE_MATCH_IN_SCALE_WEIGHT_FRACTION = "scale_match_in_scale_weight_fraction"
    SCALE_MATCH_OUT_OF_SCALE_WEIGHT_FRACTION = "scale_match_out_of_scale_weight_fraction"
    SCALE_MATCH_EXPLAINED_OUT_OF_SCALE_WEIGHT_FRACTION = "scale_match_explained_out_of_scale_weight_fraction"
    SCALE_MATCH_UNEXPLAINED_OUT_OF_SCALE_WEIGHT_FRACTION = "scale_match_unexplained_out_of_scale_weight_fraction"
    SCALE_MATCH_BEST_MARGIN = "scale_match_best_margin"
    SCALE_MATCH_OBSERVED_PITCH_CLASS_COUNT = "scale_match_observed_pitch_class_count"
    SCALE_MATCH_EXPLANATION_PITCH_CLASS_COUNT = "scale_match_explanation_pitch_class_count"
    SCALE_MATCH_SUPPORT_CANDIDATE_COUNT = "scale_match_support_candidate_count"
    SCALE_MATCH_TIED_BEST_CANDIDATE_COUNT = "scale_match_tied_best_candidate_count"
    SCALE_MATCH_DECLARED_MATCH_USED = "scale_match_declared_match_used"
    SCALE_MATCH_LOW_CONFIDENCE = "scale_match_low_confidence"
    SCALE_MATCH_AMBIGUOUS = "scale_match_ambiguous"
    SCALE_MATCH_NO_PITCHES = "scale_match_no_pitches"
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
    ACCIDENTAL_NOTE_FRACTION = "accidental_note_fraction"
    IN_SCALE_NOTE_FRACTION = "in_scale_note_fraction"
    NOTE_DENSITY_PER_BEAT = "note_density_per_beat"
    ONSET_DENSITY_PER_BEAT = "onset_density_per_beat"
    RIGHT_ONSET_DENSITY_PER_BEAT = "right_onset_density_per_beat"
    LEFT_ONSET_DENSITY_PER_BEAT = "left_onset_density_per_beat"
    SHORTEST_NOTE_DURATION_BEATS = "shortest_note_duration_beats"
    HAS_DOTTED_NOTES = "has_dotted_notes"
    MAX_NOTES_PER_ONSET = "max_notes_per_onset"
    MAX_NOTES_PER_HAND = "max_notes_per_hand"
    MAX_ONSET_SPAN_SEMITONES = "max_onset_span_semitones"
    MAX_MELODIC_GAP_SEMITONES = "max_melodic_gap_semitones"
    STATIC_HAND_SPAN_DEGREES = "static_hand_span_degrees"
    SYNCHRONIZED_ONSET_FRACTION = "synchronized_onset_fraction"
    INDEPENDENT_ONSET_FRACTION = "independent_onset_fraction"


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
    ParsedManifestField.DECLARED_KEY_FIFTHS,
    ParsedManifestField.TIME_SIGNATURE,
)

ENCODED_MANIFEST_FIELDS: Final[tuple[EncodedManifestField, ...]] = (
    EncodedManifestField.SEGMENT_ID,
    EncodedManifestField.SOURCE_ID,
    EncodedManifestField.SOURCE_PATH,
    EncodedManifestField.PARSED_PATH,
    EncodedManifestField.ENCODED_SHARD,
    EncodedManifestField.ENCODED_LINE,
    EncodedManifestField.SEGMENTATION_MODE,
    EncodedManifestField.WINDOW_START_BAR,
    EncodedManifestField.BAR_COUNT,
    EncodedManifestField.TOKEN_COUNT,
    EncodedManifestField.ELIGIBLE_FOR_TRAINING,
    EncodedManifestField.INELIGIBILITY_REASONS,
    EncodedManifestField.SCALE_ROOT,
    EncodedManifestField.SCALE_TYPE,
    EncodedManifestField.DECLARED_KEY_FIFTHS,
    EncodedManifestField.SPELLING_KEY_FIFTHS,
    EncodedManifestField.SPELLING_CONTEXT_SOURCE,
    EncodedManifestField.SCALE_MATCH_IN_SCALE_WEIGHT_FRACTION,
    EncodedManifestField.SCALE_MATCH_OUT_OF_SCALE_WEIGHT_FRACTION,
    EncodedManifestField.SCALE_MATCH_EXPLAINED_OUT_OF_SCALE_WEIGHT_FRACTION,
    EncodedManifestField.SCALE_MATCH_UNEXPLAINED_OUT_OF_SCALE_WEIGHT_FRACTION,
    EncodedManifestField.SCALE_MATCH_BEST_MARGIN,
    EncodedManifestField.SCALE_MATCH_OBSERVED_PITCH_CLASS_COUNT,
    EncodedManifestField.SCALE_MATCH_EXPLANATION_PITCH_CLASS_COUNT,
    EncodedManifestField.SCALE_MATCH_SUPPORT_CANDIDATE_COUNT,
    EncodedManifestField.SCALE_MATCH_TIED_BEST_CANDIDATE_COUNT,
    EncodedManifestField.SCALE_MATCH_DECLARED_MATCH_USED,
    EncodedManifestField.SCALE_MATCH_LOW_CONFIDENCE,
    EncodedManifestField.SCALE_MATCH_AMBIGUOUS,
    EncodedManifestField.SCALE_MATCH_NO_PITCHES,
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
    EncodedManifestField.ACCIDENTAL_NOTE_FRACTION,
    EncodedManifestField.IN_SCALE_NOTE_FRACTION,
    EncodedManifestField.NOTE_DENSITY_PER_BEAT,
    EncodedManifestField.ONSET_DENSITY_PER_BEAT,
    EncodedManifestField.RIGHT_ONSET_DENSITY_PER_BEAT,
    EncodedManifestField.LEFT_ONSET_DENSITY_PER_BEAT,
    EncodedManifestField.SHORTEST_NOTE_DURATION_BEATS,
    EncodedManifestField.HAS_DOTTED_NOTES,
    EncodedManifestField.MAX_NOTES_PER_ONSET,
    EncodedManifestField.MAX_NOTES_PER_HAND,
    EncodedManifestField.MAX_ONSET_SPAN_SEMITONES,
    EncodedManifestField.MAX_MELODIC_GAP_SEMITONES,
    EncodedManifestField.STATIC_HAND_SPAN_DEGREES,
    EncodedManifestField.SYNCHRONIZED_ONSET_FRACTION,
    EncodedManifestField.INDEPENDENT_ONSET_FRACTION,
)


def write_parsed_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    _write_manifest_csv(rows, path, fieldnames=_field_values(PARSED_MANIFEST_FIELDS))


def write_encoded_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    _write_manifest_csv(rows, path, fieldnames=_field_values(ENCODED_MANIFEST_FIELDS))


def read_parsed_manifest(path: Path) -> list[dict[str, str]]:
    return list(_iter_manifest_csv(path))


def read_encoded_manifest(path: Path) -> list[dict[str, str]]:
    return list(iter_encoded_manifest(path))


def iter_parsed_manifest(path: Path) -> Iterator[dict[str, str]]:
    yield from _iter_manifest_csv(path)


def iter_encoded_manifest(path: Path) -> Iterator[dict[str, str]]:
    yield from _iter_manifest_csv(path)


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
        ParsedManifestField.DECLARED_KEY_FIFTHS: (
            score.declared_key_fifths if score.declared_key_fifths is not None else ""
        ),
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
        ParsedManifestField.DECLARED_KEY_FIFTHS: "",
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
    diagnostics: SegmentDiagnostics,
    encoded_sample: "EncodedExercise | None",
    encoded_shard: Path,
    encoded_line: int | None,
    segmentation_mode: SegmentationMode,
) -> dict[str, Any]:
    scale_match = segment.metadata.scale_match
    tokenization_context = segment.metadata.tokenization_context
    declared_key_fifths = _declared_key_fifths(segment)
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
        EncodedManifestField.SEGMENTATION_MODE: segmentation_mode.value,
        EncodedManifestField.WINDOW_START_BAR: segment.metadata.window_start_bar,
        EncodedManifestField.BAR_COUNT: segment.metadata.bar_count,
        EncodedManifestField.TOKEN_COUNT: len(segment.tokens),
        EncodedManifestField.ELIGIBLE_FOR_TRAINING: segment.metadata.eligible_for_training,
        EncodedManifestField.INELIGIBILITY_REASONS: "|".join(
            sorted(reason.value for reason in segment.metadata.ineligibility_reasons)
        ),
        EncodedManifestField.SCALE_ROOT: segment.metadata.scale_root,
        EncodedManifestField.SCALE_TYPE: segment.metadata.scale_type.value,
        EncodedManifestField.DECLARED_KEY_FIFTHS: declared_key_fifths if declared_key_fifths is not None else "",
        EncodedManifestField.SPELLING_KEY_FIFTHS: (
            tokenization_context.spelling_key_fifths if tokenization_context is not None else ""
        ),
        EncodedManifestField.SPELLING_CONTEXT_SOURCE: (
            tokenization_context.spelling_context_source.value if tokenization_context is not None else ""
        ),
        EncodedManifestField.SCALE_MATCH_IN_SCALE_WEIGHT_FRACTION: (
            scale_match.in_scale_weight_fraction if scale_match is not None else ""
        ),
        EncodedManifestField.SCALE_MATCH_OUT_OF_SCALE_WEIGHT_FRACTION: (
            scale_match.out_of_scale_weight_fraction if scale_match is not None else ""
        ),
        EncodedManifestField.SCALE_MATCH_EXPLAINED_OUT_OF_SCALE_WEIGHT_FRACTION: (
            scale_match.explained_out_of_scale_weight_fraction if scale_match is not None else ""
        ),
        EncodedManifestField.SCALE_MATCH_UNEXPLAINED_OUT_OF_SCALE_WEIGHT_FRACTION: (
            scale_match.unexplained_out_of_scale_weight_fraction if scale_match is not None else ""
        ),
        EncodedManifestField.SCALE_MATCH_BEST_MARGIN: scale_match.best_margin if scale_match is not None else "",
        EncodedManifestField.SCALE_MATCH_OBSERVED_PITCH_CLASS_COUNT: (
            scale_match.observed_pitch_class_count if scale_match is not None else ""
        ),
        EncodedManifestField.SCALE_MATCH_EXPLANATION_PITCH_CLASS_COUNT: (
            scale_match.explanation_pitch_class_count if scale_match is not None else ""
        ),
        EncodedManifestField.SCALE_MATCH_SUPPORT_CANDIDATE_COUNT: (
            scale_match.support_candidate_count if scale_match is not None else ""
        ),
        EncodedManifestField.SCALE_MATCH_TIED_BEST_CANDIDATE_COUNT: (
            scale_match.tied_best_candidate_count if scale_match is not None else ""
        ),
        EncodedManifestField.SCALE_MATCH_DECLARED_MATCH_USED: (
            scale_match.declared_match_used if scale_match is not None else ""
        ),
        EncodedManifestField.SCALE_MATCH_LOW_CONFIDENCE: scale_match.low_confidence if scale_match is not None else "",
        EncodedManifestField.SCALE_MATCH_AMBIGUOUS: scale_match.ambiguous if scale_match is not None else "",
        EncodedManifestField.SCALE_MATCH_NO_PITCHES: scale_match.no_pitches if scale_match is not None else "",
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
        EncodedManifestField.ACCIDENTAL_NOTE_FRACTION: diagnostics.accidental_note_fraction,
        EncodedManifestField.IN_SCALE_NOTE_FRACTION: diagnostics.in_scale_note_fraction,
        EncodedManifestField.NOTE_DENSITY_PER_BEAT: diagnostics.note_density_per_beat,
        EncodedManifestField.ONSET_DENSITY_PER_BEAT: diagnostics.onset_density_per_beat,
        EncodedManifestField.RIGHT_ONSET_DENSITY_PER_BEAT: diagnostics.right_onset_density_per_beat,
        EncodedManifestField.LEFT_ONSET_DENSITY_PER_BEAT: diagnostics.left_onset_density_per_beat,
        EncodedManifestField.SHORTEST_NOTE_DURATION_BEATS: diagnostics.shortest_note_duration_beats,
        EncodedManifestField.HAS_DOTTED_NOTES: diagnostics.has_dotted_notes,
        EncodedManifestField.MAX_NOTES_PER_ONSET: diagnostics.max_notes_per_onset,
        EncodedManifestField.MAX_NOTES_PER_HAND: diagnostics.max_notes_per_hand,
        EncodedManifestField.MAX_ONSET_SPAN_SEMITONES: diagnostics.max_onset_span_semitones,
        EncodedManifestField.MAX_MELODIC_GAP_SEMITONES: diagnostics.max_melodic_gap_semitones,
        EncodedManifestField.STATIC_HAND_SPAN_DEGREES: diagnostics.static_hand_span_degrees,
        EncodedManifestField.SYNCHRONIZED_ONSET_FRACTION: diagnostics.synchronized_onset_fraction,
        EncodedManifestField.INDEPENDENT_ONSET_FRACTION: diagnostics.independent_onset_fraction,
    }


def _declared_key_fifths(segment: Segment) -> int | None:
    if segment.metadata.scale_match is not None and segment.metadata.scale_match.declared_key_fifths is not None:
        return segment.metadata.scale_match.declared_key_fifths

    if (
        segment.metadata.tokenization_context is not None
        and segment.metadata.tokenization_context.declared_key_fifths is not None
    ):
        return segment.metadata.tokenization_context.declared_key_fifths

    return None


def _field_values(fields: tuple[StrEnum, ...]) -> tuple[str, ...]:
    return tuple(field.value for field in fields)


def _write_manifest_csv(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    fieldnames: tuple[str, ...],
) -> None:
    write_csv_rows(path, columns=fieldnames, rows=rows)


def _iter_manifest_csv(path: Path) -> Iterator[dict[str, str]]:
    if not path.exists():
        return

    with path.open("r", newline="", encoding="utf-8") as file:
        yield from csv.DictReader(file)


def _relative_text(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()
