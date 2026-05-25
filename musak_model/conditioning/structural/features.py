from __future__ import annotations

from fractions import Fraction

from pydantic import BaseModel, ConfigDict

from musak_model.conditioning.structural.schema import StructuralControlFeatures
from musak_model.data.schema import Segment
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.pitch import note_token_to_midi_pitch, note_token_to_static_hand_position
from musak_model.tokens.schema import Hand, HandToken, JoinWithPreviousToken, NoteToken
from musak_shared.elements import is_dotted_duration


def extract_structural_control_features(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
) -> StructuralControlFeatures:
    state = _FeatureState()
    tokens = segment.tokens
    active_hand = Hand.RIGHT

    for index, token in enumerate(tokens):
        match token:
            case HandToken():
                active_hand = token.hand
            case NoteToken():
                next_token = tokens[index + 1] if index + 1 < len(tokens) else None
                joins_previous_onset = isinstance(next_token, JoinWithPreviousToken)
                state = state.add_note(
                    token,
                    hand=active_hand,
                    joins_previous_onset=joins_previous_onset,
                    segment=segment,
                    duration_vocabulary=duration_vocabulary,
                )
            case JoinWithPreviousToken():
                continue
            case _:
                continue

    return state.to_features()


class _FeatureState(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    shortest_note_duration: Fraction | None = None
    has_dotted_notes: bool = False
    max_notes_per_onset: int = 0
    right_max_notes_per_hand: int = 0
    left_max_notes_per_hand: int = 0
    max_onset_span_semitones: int = 0
    max_melodic_gap_semitones: int = 0
    current_onset_count: int = 0
    right_static_positions: tuple[int, ...] = ()
    left_static_positions: tuple[int, ...] = ()
    right_last_onset_pitches: tuple[int, ...] = ()
    left_last_onset_pitches: tuple[int, ...] = ()
    right_current_onset_count: int = 0
    left_current_onset_count: int = 0

    def add_note(
        self,
        token: NoteToken,
        *,
        hand: Hand,
        joins_previous_onset: bool,
        segment: Segment,
        duration_vocabulary: DurationVocabulary,
    ) -> _FeatureState:
        duration = duration_vocabulary.id_to_fraction(token.duration_id)
        midi_pitch = note_token_to_midi_pitch(
            token,
            scale_root=segment.scale_root,
            scale_type=segment.scale_type,
            hand=hand,
        )
        state = self.model_copy(
            update={
                "shortest_note_duration": _minimum_optional_fraction(self.shortest_note_duration, duration),
                "has_dotted_notes": self.has_dotted_notes or is_dotted_duration(duration),
            }
        )
        state = state.add_static_position(note_token_to_static_hand_position(token), hand=hand)
        if joins_previous_onset:
            return state.add_chord_note(midi_pitch, hand=hand)

        return state.add_new_onset(midi_pitch, hand=hand)

    def to_features(self) -> StructuralControlFeatures:
        return StructuralControlFeatures(
            shortest_note_duration=self.shortest_note_duration,
            has_dotted_notes=self.has_dotted_notes,
            max_notes_per_onset=self.max_notes_per_onset,
            max_notes_per_hand=max(self.right_max_notes_per_hand, self.left_max_notes_per_hand),
            max_onset_span_semitones=self.max_onset_span_semitones,
            max_melodic_gap_semitones=self.max_melodic_gap_semitones,
            static_hand_span_degrees=max(
                _inclusive_span(self.right_static_positions),
                _inclusive_span(self.left_static_positions),
            ),
            bar_count=None,
        )

    def add_static_position(self, position: int, *, hand: Hand) -> _FeatureState:
        match hand:
            case Hand.RIGHT:
                return self.model_copy(update={"right_static_positions": (*self.right_static_positions, position)})
            case Hand.LEFT:
                return self.model_copy(update={"left_static_positions": (*self.left_static_positions, position)})

    def add_chord_note(self, midi_pitch: int, *, hand: Hand) -> _FeatureState:
        match hand:
            case Hand.RIGHT:
                count = self.right_current_onset_count + 1
                onset_count = self.current_onset_count + 1
                current_pitches = (*self.right_last_onset_pitches, midi_pitch)
                return self.model_copy(
                    update={
                        "right_current_onset_count": count,
                        "right_max_notes_per_hand": max(self.right_max_notes_per_hand, count),
                        "current_onset_count": onset_count,
                        "right_last_onset_pitches": current_pitches,
                        "max_notes_per_onset": max(self.max_notes_per_onset, onset_count),
                        "max_onset_span_semitones": max(
                            self.max_onset_span_semitones,
                            _distance_span(current_pitches),
                        ),
                    }
                )
            case Hand.LEFT:
                count = self.left_current_onset_count + 1
                onset_count = self.current_onset_count + 1
                current_pitches = (*self.left_last_onset_pitches, midi_pitch)
                return self.model_copy(
                    update={
                        "left_current_onset_count": count,
                        "left_max_notes_per_hand": max(self.left_max_notes_per_hand, count),
                        "current_onset_count": onset_count,
                        "left_last_onset_pitches": current_pitches,
                        "max_notes_per_onset": max(self.max_notes_per_onset, onset_count),
                        "max_onset_span_semitones": max(
                            self.max_onset_span_semitones,
                            _distance_span(current_pitches),
                        ),
                    }
                )

    def add_new_onset(self, midi_pitch: int, *, hand: Hand) -> _FeatureState:
        match hand:
            case Hand.RIGHT:
                return self.model_copy(
                    update={
                        "right_current_onset_count": 1,
                        "right_max_notes_per_hand": max(self.right_max_notes_per_hand, 1),
                        "current_onset_count": 1,
                        "right_last_onset_pitches": (midi_pitch,),
                        "max_notes_per_onset": max(self.max_notes_per_onset, 1),
                        "max_melodic_gap_semitones": max(
                            self.max_melodic_gap_semitones,
                            _melodic_gap(midi_pitch, self.right_last_onset_pitches),
                        ),
                    }
                )
            case Hand.LEFT:
                return self.model_copy(
                    update={
                        "left_current_onset_count": 1,
                        "left_max_notes_per_hand": max(self.left_max_notes_per_hand, 1),
                        "current_onset_count": 1,
                        "left_last_onset_pitches": (midi_pitch,),
                        "max_notes_per_onset": max(self.max_notes_per_onset, 1),
                        "max_melodic_gap_semitones": max(
                            self.max_melodic_gap_semitones,
                            _melodic_gap(midi_pitch, self.left_last_onset_pitches),
                        ),
                    }
                )


def _minimum_optional_fraction(current: Fraction | None, candidate: Fraction) -> Fraction:
    if current is None:
        return candidate

    return min(current, candidate)


def _melodic_gap(midi_pitch: int, previous_onset_pitches: tuple[int, ...]) -> int:
    if not previous_onset_pitches:
        return 0

    return min(abs(midi_pitch - previous_pitch) for previous_pitch in previous_onset_pitches)


def _inclusive_span(values: tuple[int, ...]) -> int:
    if len(values) < 2:
        return 0

    return max(values) - min(values) + 1


def _distance_span(values: tuple[int, ...]) -> int:
    if len(values) < 2:
        return 0

    return max(values) - min(values)
