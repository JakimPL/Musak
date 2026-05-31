from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from musak_model.data.schema import Segment
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.pitch import note_token_to_midi_pitch
from musak_model.tokens.schema import BarToken, EndToken, Hand, HandToken, NoteToken, ScaleType
from musak_shared.elements import is_dotted_duration


@dataclass(frozen=True)
class MusicalAuxiliaryFeatures:
    note_density: float
    rhythmic_diversity: float
    voice_independence: float
    uses_accidentals: bool
    dotted_duration: bool
    hand_span: int


@dataclass
class _FeatureAccumulator:
    note_count: int = 0
    duration_ids: set[int] = field(default_factory=set)
    right_rhythm: list[Fraction] = field(default_factory=list)
    left_rhythm: list[Fraction] = field(default_factory=list)
    right_pitches: list[int] = field(default_factory=list)
    left_pitches: list[int] = field(default_factory=list)
    uses_accidentals: bool = False
    dotted_duration: bool = False

    def add_note(
        self,
        token: NoteToken,
        *,
        hand: Hand,
        scale_root: int,
        scale_type: ScaleType,
        duration_vocabulary: DurationVocabulary,
    ) -> None:
        duration = duration_vocabulary.id_to_fraction(token.duration_id)
        midi_pitch = note_token_to_midi_pitch(
            token,
            scale_root=scale_root,
            scale_type=scale_type,
            hand=hand,
        )
        self.note_count += 1
        self.duration_ids.add(token.duration_id)
        self.uses_accidentals = self.uses_accidentals or token.accidental != 0
        self.dotted_duration = self.dotted_duration or is_dotted_duration(duration)

        rhythm = self.right_rhythm if hand == Hand.RIGHT else self.left_rhythm
        rhythm.append(duration)
        pitches = self.right_pitches if hand == Hand.RIGHT else self.left_pitches
        pitches.append(midi_pitch)

    @property
    def hand_span(self) -> int:
        spans = [
            max(pitches) - min(pitches) for pitches in (self.right_pitches, self.left_pitches) if len(pitches) >= 2
        ]
        return max(spans) if spans else 0


def musical_auxiliary_features_from_segment(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
) -> MusicalAuxiliaryFeatures:
    bar_accumulators = bar_musical_auxiliary_feature_accumulators(
        segment,
        duration_vocabulary=duration_vocabulary,
    )
    combined = _combine_accumulators(bar_accumulators)
    return _features_from_accumulator(
        combined,
        duration_vocabulary=duration_vocabulary,
        beat_count=segment.time_numerator * len(bar_accumulators),
        hand_span=max((accumulator.hand_span for accumulator in bar_accumulators), default=0),
    )


def bar_musical_auxiliary_features_from_segment(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
) -> tuple[MusicalAuxiliaryFeatures, ...]:
    return tuple(
        _features_from_accumulator(
            accumulator,
            duration_vocabulary=duration_vocabulary,
            beat_count=segment.time_numerator,
            hand_span=accumulator.hand_span,
        )
        for accumulator in bar_musical_auxiliary_feature_accumulators(
            segment,
            duration_vocabulary=duration_vocabulary,
        )
    )


def bar_musical_auxiliary_feature_accumulators(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
) -> tuple[_FeatureAccumulator, ...]:
    accumulators = [_FeatureAccumulator() for _ in range(_initial_bar_count(segment))]
    active_hand = Hand.RIGHT
    bar_index = 0

    for token in segment.tokens:
        if isinstance(token, EndToken):
            break

        if isinstance(token, HandToken):
            active_hand = token.hand
            continue

        if isinstance(token, BarToken):
            bar_index += 1
            continue

        if isinstance(token, NoteToken):
            _ensure_bar_accumulator(accumulators, bar_index=bar_index)
            accumulators[bar_index].add_note(
                token,
                hand=active_hand,
                scale_root=segment.scale_root,
                scale_type=segment.scale_type,
                duration_vocabulary=duration_vocabulary,
            )

    return tuple(accumulators)


def _initial_bar_count(segment: Segment) -> int:
    if segment.bar_count > 0:
        return segment.bar_count

    return 1 if segment.tokens else 0


def _ensure_bar_accumulator(accumulators: list[_FeatureAccumulator], *, bar_index: int) -> None:
    while bar_index >= len(accumulators):
        accumulators.append(_FeatureAccumulator())


def _combine_accumulators(accumulators: tuple[_FeatureAccumulator, ...]) -> _FeatureAccumulator:
    combined = _FeatureAccumulator()
    for accumulator in accumulators:
        combined.note_count += accumulator.note_count
        combined.duration_ids.update(accumulator.duration_ids)
        combined.right_rhythm.extend(accumulator.right_rhythm)
        combined.left_rhythm.extend(accumulator.left_rhythm)
        combined.right_pitches.extend(accumulator.right_pitches)
        combined.left_pitches.extend(accumulator.left_pitches)
        combined.uses_accidentals = combined.uses_accidentals or accumulator.uses_accidentals
        combined.dotted_duration = combined.dotted_duration or accumulator.dotted_duration

    return combined


def _features_from_accumulator(
    accumulator: _FeatureAccumulator,
    *,
    duration_vocabulary: DurationVocabulary,
    beat_count: int,
    hand_span: int,
) -> MusicalAuxiliaryFeatures:
    return MusicalAuxiliaryFeatures(
        note_density=_note_density(accumulator.note_count, beat_count=beat_count),
        rhythmic_diversity=len(accumulator.duration_ids) / duration_vocabulary.vocabulary_size(),
        voice_independence=_voice_independence(accumulator.right_rhythm, accumulator.left_rhythm),
        uses_accidentals=accumulator.uses_accidentals,
        dotted_duration=accumulator.dotted_duration,
        hand_span=hand_span,
    )


def _note_density(note_count: int, *, beat_count: int) -> float:
    if beat_count <= 0:
        return 0.0

    return note_count / beat_count


def _voice_independence(right_rhythm: list[Fraction], left_rhythm: list[Fraction]) -> float:
    if len(right_rhythm) != len(left_rhythm) or not right_rhythm:
        return 0.0

    matching = sum(
        1 for right_duration, left_duration in zip(right_rhythm, left_rhythm) if right_duration == left_duration
    )
    return 1.0 - matching / len(right_rhythm)
