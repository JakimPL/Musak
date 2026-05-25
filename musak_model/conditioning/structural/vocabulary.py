from fractions import Fraction

from musak_model.conditioning.structural.config import StructuralConditioningConfig
from musak_model.conditioning.structural.constants import (
    BOOLEAN_CONTROL_VOCABULARY_SIZE,
    FALSE_CONTROL_ID,
    STRUCTURAL_CONTROL_ORDER,
    TRUE_CONTROL_ID,
    UNKNOWN_CONTROL_ID,
    StructuralControlName,
)
from musak_model.conditioning.structural.schema import StructuralControlFeatures


class StructuralControlVocabulary:
    def __init__(self, config: StructuralConditioningConfig) -> None:
        self._config = config

    @property
    def vocabulary_sizes(self) -> tuple[int, ...]:
        return tuple(self.vocabulary_size(control_name) for control_name in STRUCTURAL_CONTROL_ORDER)

    def control_index(self, control_name: StructuralControlName) -> int:
        return STRUCTURAL_CONTROL_ORDER.index(control_name)

    def vocabulary_size(self, control_name: StructuralControlName) -> int:
        match control_name:
            case StructuralControlName.SHORTEST_NOTE_DURATION:
                return _bucket_size(self._config.shortest_note_duration.duration_thresholds)
            case StructuralControlName.HAS_DOTTED_NOTES:
                return BOOLEAN_CONTROL_VOCABULARY_SIZE
            case StructuralControlName.MAX_NOTES_PER_ONSET:
                return _bucket_size(self._config.max_notes_per_onset.thresholds)
            case StructuralControlName.MAX_NOTES_PER_HAND:
                return _bucket_size(self._config.max_notes_per_hand.thresholds)
            case StructuralControlName.MAX_ONSET_SPAN_SEMITONES:
                return _bucket_size(self._config.max_onset_span_semitones.thresholds)
            case StructuralControlName.MAX_MELODIC_GAP_SEMITONES:
                return _bucket_size(self._config.max_melodic_gap_semitones.thresholds)
            case StructuralControlName.STATIC_HAND_SPAN_DEGREES:
                return _bucket_size(self._config.static_hand_span_degrees.thresholds)
            case StructuralControlName.BAR_COUNT:
                return _bucket_size(self._config.bar_count.thresholds)

    def features_to_ids(self, features: StructuralControlFeatures | None) -> tuple[int, ...]:
        if features is None:
            return tuple(UNKNOWN_CONTROL_ID for _ in STRUCTURAL_CONTROL_ORDER)

        return (
            _fraction_bucket_id(
                features.shortest_note_duration,
                self._config.shortest_note_duration.duration_thresholds,
            ),
            _boolean_control_id(features.has_dotted_notes),
            _integer_bucket_id(features.max_notes_per_onset, self._config.max_notes_per_onset.thresholds),
            _integer_bucket_id(features.max_notes_per_hand, self._config.max_notes_per_hand.thresholds),
            _integer_bucket_id(features.max_onset_span_semitones, self._config.max_onset_span_semitones.thresholds),
            _integer_bucket_id(features.max_melodic_gap_semitones, self._config.max_melodic_gap_semitones.thresholds),
            _integer_bucket_id(features.static_hand_span_degrees, self._config.static_hand_span_degrees.thresholds),
            _integer_bucket_id(features.bar_count, self._config.bar_count.thresholds),
        )


def _bucket_size(thresholds: tuple[object, ...]) -> int:
    return len(thresholds) + 2


def _integer_bucket_id(value: int | None, thresholds: tuple[int, ...]) -> int:
    if value is None:
        return UNKNOWN_CONTROL_ID

    for index, threshold in enumerate(thresholds, start=1):
        if value <= threshold:
            return index

    return len(thresholds) + 1


def _boolean_control_id(value: bool | None) -> int:
    if value is None:
        return UNKNOWN_CONTROL_ID

    return TRUE_CONTROL_ID if value else FALSE_CONTROL_ID


def _fraction_bucket_id(value: Fraction | None, thresholds: tuple[Fraction, ...]) -> int:
    if value is None:
        return UNKNOWN_CONTROL_ID

    for index, threshold in enumerate(thresholds, start=1):
        if value <= threshold:
            return index

    return len(thresholds) + 1
