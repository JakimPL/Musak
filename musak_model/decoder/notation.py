from __future__ import annotations

from fractions import Fraction
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from musak_model.data.schema import Segment
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.pitch import note_token_to_midi_pitch
from musak_model.tokens.schema import (
    BarToken,
    EndToken,
    Hand,
    HandToken,
    HoldToken,
    JoinWithPreviousToken,
    NoteToken,
    RestToken,
    ScaleType,
    StartToken,
)
from musak_shared.elements import MIDI_MAX_PITCH
from musak_shared.names import midi_to_vexflow_key
from musak_shared.notation.schema import (
    EIGHTH,
    HALF,
    QUARTER,
    REST_SUFFIX,
    SIXTEENTH,
    THIRTY_SECOND,
    WHOLE,
    Clef,
    NoteData,
    ScoreData,
    StaveData,
    VexflowDuration,
    VoiceData,
)

_POWER_DURATION_BY_FRACTION: Final[dict[Fraction, str]] = {
    Fraction(1, 1): WHOLE,
    Fraction(1, 2): HALF,
    Fraction(1, 4): QUARTER,
    Fraction(1, 8): EIGHTH,
    Fraction(1, 16): SIXTEENTH,
    Fraction(1, 32): THIRTY_SECOND,
}
_MAX_DOTS: Final[int] = 2
_REST_KEY: Final[str] = "b/4"
_NOTE_KIND: Final[Literal["note"]] = "note"
_REST_KIND: Final[Literal["rest"]] = "rest"
_MAJOR_KEY_SIGNATURES_BY_FIFTHS: Final[dict[int, str]] = {
    -7: "Cb",
    -6: "Gb",
    -5: "Db",
    -4: "Ab",
    -3: "Eb",
    -2: "Bb",
    -1: "F",
    0: "C",
    1: "G",
    2: "D",
    3: "A",
    4: "E",
    5: "B",
    6: "F#",
    7: "C#",
}
_PARENT_MAJOR_OFFSET_BY_SCALE_TYPE: Final[dict[ScaleType, int]] = {
    ScaleType.MAJOR: 0,
    ScaleType.DORIAN: -2,
    ScaleType.PHRYGIAN: -4,
    ScaleType.LYDIAN: -5,
    ScaleType.MIXOLYDIAN: -7,
    ScaleType.AEOLIAN: -9,
    ScaleType.HARMONIC_MINOR: -9,
    ScaleType.MELODIC_MINOR: -9,
    ScaleType.LOCRIAN: -11,
}
_REPRESENTABLE_DURATIONS: Final[tuple[Fraction, ...]] = tuple(
    sorted(
        {
            duration * Fraction(2 ** (dots + 1) - 1, 2**dots)
            for duration in _POWER_DURATION_BY_FRACTION
            for dots in range(_MAX_DOTS + 1)
        },
        reverse=True,
    )
)


class UnsupportedNotationDurationError(ValueError):
    """Raised when a decoded duration cannot be represented by the current VexFlow schema."""


class DecodedNotationEvent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    hand: Hand
    midi_pitch: int = Field(ge=0, le=MIDI_MAX_PITCH)
    vexflow_key: str
    start: Fraction = Field(ge=0)
    duration: Fraction = Field(gt=0)

    @property
    def end(self) -> Fraction:
        return self.start + self.duration


class ModelNotationEvent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    kind: Literal["note", "rest"]
    start: Fraction = Field(ge=0)
    duration: Fraction = Field(gt=0)
    midi_pitches: tuple[int, ...] = ()
    vexflow_keys: tuple[str, ...] = ()
    tie_start: bool = False
    tie_stop: bool = False

    @property
    def end(self) -> Fraction:
        return self.start + self.duration


def segment_to_score_data(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
    tempo: int | None = None,
    measures_per_row: int | None = None,
    max_bars: int | None = None,
) -> ScoreData:
    events = tuple(segment_to_notation_events(segment, duration_vocabulary=duration_vocabulary))
    measure_duration = Fraction(segment.time_numerator, segment.time_denominator)
    bar_count = max(segment.bar_count, _token_bar_count(segment))
    displayed_bar_count = bar_count if max_bars is None else min(bar_count, max_bars)
    key_signature = key_signature_name(key_root=segment.key_root, scale_type=segment.scale_type)
    rows: list[list[StaveData]] = []
    for first_measure in range(0, displayed_bar_count, measures_per_row or max(displayed_bar_count, 1)):
        last_measure = min(first_measure + (measures_per_row or displayed_bar_count), displayed_bar_count)
        rows.extend(
            _score_rows_for_measure_range(
                events,
                measure_duration=measure_duration,
                first_measure=first_measure,
                last_measure=last_measure,
                key_signature=key_signature,
                time_signature=(segment.time_numerator, segment.time_denominator),
            )
        )

    return ScoreData(rows=rows, tempo=tempo, max_notes_per_measure=_max_notes_per_measure(rows))


def segment_to_notation_events(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
    default_hand: Hand = Hand.RIGHT,
) -> list[DecodedNotationEvent]:
    measure_duration = Fraction(segment.time_numerator, segment.time_denominator)
    active_hand = default_hand
    bar_index = 0
    cursors = {Hand.RIGHT: Fraction(0), Hand.LEFT: Fraction(0)}
    last_attack_indices: dict[Hand, list[int]] = {Hand.RIGHT: [], Hand.LEFT: []}
    events: list[DecodedNotationEvent] = []

    for token in segment.tokens:
        if isinstance(token, HandToken):
            active_hand = token.hand
            continue

        if isinstance(token, StartToken):
            continue

        if isinstance(token, BarToken):
            bar_index += 1
            cursors = {Hand.RIGHT: Fraction(0), Hand.LEFT: Fraction(0)}
            continue

        if isinstance(token, EndToken):
            break

        if isinstance(token, RestToken):
            cursors[active_hand] += duration_vocabulary.id_to_fraction(token.duration_id)
            continue

        if isinstance(token, HoldToken):
            duration = duration_vocabulary.id_to_fraction(token.duration_id)
            _extend_last_attack(
                events,
                event_indices=last_attack_indices[active_hand],
                duration=duration,
                hand=active_hand,
            )
            cursors[active_hand] += duration
            continue

        if isinstance(token, NoteToken):
            duration = duration_vocabulary.id_to_fraction(token.duration_id)
            midi_pitch = note_token_to_midi_pitch(
                token,
                key_root=segment.key_root,
                scale_type=segment.scale_type,
                hand=active_hand,
            )
            events.append(
                DecodedNotationEvent(
                    hand=active_hand,
                    midi_pitch=midi_pitch,
                    vexflow_key=midi_to_vexflow_key(midi_pitch, prefer_flats=token.accidental < 0),
                    start=bar_index * measure_duration + cursors[active_hand],
                    duration=duration,
                )
            )
            cursors[active_hand] += duration
            last_attack_indices[active_hand] = [len(events) - 1]
            continue

        if isinstance(token, JoinWithPreviousToken):
            if len(events) < 2:
                raise ValueError("join-with-previous token needs at least two decoded notes")

            previous_event = events[-1]
            joined_start = events[-2].start
            events[-1] = previous_event.model_copy(update={"start": joined_start})
            last_attack_indices[active_hand] = _same_onset_event_indices(
                events,
                hand=active_hand,
                start=joined_start,
            )
            joined_end = max(events[index].end for index in last_attack_indices[active_hand])
            cursors[active_hand] = joined_end - bar_index * measure_duration
            continue

    return events


def key_signature_name(*, key_root: int, scale_type: ScaleType) -> str:
    parent_major_root = (key_root + _PARENT_MAJOR_OFFSET_BY_SCALE_TYPE[scale_type]) % 12
    return _MAJOR_KEY_SIGNATURES_BY_FIFTHS[_major_fifths_for_key_root(parent_major_root)]


def _major_fifths_for_key_root(key_root: int) -> int:
    for fifths in range(-7, 8):
        if (fifths * 7) % 12 == key_root:
            return fifths

    raise ValueError(f"cannot derive key signature for key root {key_root}")


def _extend_last_attack(
    events: list[DecodedNotationEvent],
    *,
    event_indices: list[int],
    duration: Fraction,
    hand: Hand,
) -> None:
    if not event_indices:
        raise ValueError(f"hold token needs a previous {hand.value} hand note or chord")

    for event_index in event_indices:
        event = events[event_index]
        if event.hand != hand:
            raise ValueError("hold token cannot extend a note from another hand")

        events[event_index] = event.model_copy(update={"duration": event.duration + duration})


def _same_onset_event_indices(events: list[DecodedNotationEvent], *, hand: Hand, start: Fraction) -> list[int]:
    return [index for index, event in enumerate(events) if event.hand == hand and event.start == start]


def _token_bar_count(segment: Segment) -> int:
    completed_bars = sum(1 for token in segment.tokens if isinstance(token, BarToken))
    trailing_tokens_after_bar = False
    for token in reversed(segment.tokens):
        if isinstance(token, EndToken):
            continue
        if isinstance(token, BarToken):
            break
        if isinstance(token, (HandToken, JoinWithPreviousToken, NoteToken, RestToken, HoldToken)):
            trailing_tokens_after_bar = True
            break

    return completed_bars + int(trailing_tokens_after_bar)


def _score_rows_for_measure_range(
    events: tuple[DecodedNotationEvent, ...],
    *,
    measure_duration: Fraction,
    first_measure: int,
    last_measure: int,
    key_signature: str,
    time_signature: tuple[int, int],
) -> list[list[StaveData]]:
    rows: list[list[StaveData]] = []
    right_staves = _staves_for_hand(
        events,
        hand=Hand.RIGHT,
        clef="treble",
        measure_duration=measure_duration,
        first_measure=first_measure,
        last_measure=last_measure,
        key_signature=key_signature,
        time_signature=time_signature,
    )
    left_staves = _staves_for_hand(
        events,
        hand=Hand.LEFT,
        clef="bass",
        measure_duration=measure_duration,
        first_measure=first_measure,
        last_measure=last_measure,
        key_signature=key_signature,
        time_signature=time_signature,
    )
    rows.append(right_staves)
    rows.append(left_staves)

    return rows


def _staves_for_hand(
    events: tuple[DecodedNotationEvent, ...],
    *,
    hand: Hand,
    clef: Clef,
    measure_duration: Fraction,
    first_measure: int,
    last_measure: int,
    key_signature: str,
    time_signature: tuple[int, int],
) -> list[StaveData]:
    staves: list[StaveData] = []
    hand_events = tuple(event for event in events if event.hand == hand)
    for measure_index in range(first_measure, last_measure):
        notation_events = _measure_notation_events(
            hand_events,
            measure_index=measure_index,
            measure_duration=measure_duration,
        )
        staves.append(
            StaveData(
                clef=clef,
                key_signature=key_signature if measure_index == first_measure else None,
                time_signature=time_signature if measure_index == first_measure else None,
                voices=[
                    VoiceData(
                        notes=[
                            note_data for event in notation_events for note_data in _notation_event_to_note_data(event)
                        ]
                    )
                ],
            )
        )

    return staves


def _measure_notation_events(
    events: tuple[DecodedNotationEvent, ...],
    *,
    measure_index: int,
    measure_duration: Fraction,
) -> list[ModelNotationEvent]:
    measure_start = measure_index * measure_duration
    measure_end = measure_start + measure_duration
    fragments: list[ModelNotationEvent] = []
    for start, event_group in _group_events_by_onset(events):
        group_end = max(event.end for event in event_group)
        if group_end <= measure_start or start >= measure_end:
            continue

        fragment_start = max(start, measure_start)
        fragment_end = min(group_end, measure_end)
        if fragment_start >= fragment_end:
            continue

        sorted_group = tuple(sorted(event_group, key=lambda event: event.midi_pitch))
        fragments.append(
            ModelNotationEvent(
                kind=_NOTE_KIND,
                start=fragment_start - measure_start,
                duration=fragment_end - fragment_start,
                midi_pitches=tuple(event.midi_pitch for event in sorted_group),
                vexflow_keys=tuple(event.vexflow_key for event in sorted_group),
                tie_start=fragment_end < group_end,
                tie_stop=fragment_start > start,
            )
        )

    return _fill_rests(sorted(fragments, key=lambda event: (event.start, event.midi_pitches)), measure_duration)


def _group_events_by_onset(
    events: tuple[DecodedNotationEvent, ...],
) -> list[tuple[Fraction, tuple[DecodedNotationEvent, ...]]]:
    groups: dict[Fraction, list[DecodedNotationEvent]] = {}
    for event in events:
        groups.setdefault(event.start, []).append(event)

    return [(start, tuple(group)) for start, group in sorted(groups.items())]


def _fill_rests(events: list[ModelNotationEvent], measure_duration: Fraction) -> list[ModelNotationEvent]:
    output: list[ModelNotationEvent] = []
    cursor = Fraction(0)
    for event in events:
        if event.start > cursor:
            output.append(ModelNotationEvent(kind=_REST_KIND, start=cursor, duration=event.start - cursor))
        output.append(event)
        cursor = max(cursor, event.end)

    if cursor < measure_duration:
        output.append(ModelNotationEvent(kind=_REST_KIND, start=cursor, duration=measure_duration - cursor))

    return output


def _notation_event_to_note_data(event: ModelNotationEvent) -> list[NoteData]:
    fragments = _split_duration(event.duration)
    if event.kind == _REST_KIND:
        return [_rest_note_data(duration) for duration in fragments]

    notes: list[NoteData] = []
    for index, fragment in enumerate(fragments):
        duration, dots = _duration_to_vexflow(fragment)
        notes.append(
            NoteData(
                keys=list(event.vexflow_keys),
                duration=duration,
                dots=dots,
                tie_start=event.tie_start or index < len(fragments) - 1,
                tie_stop=event.tie_stop or index > 0,
            )
        )

    return notes


def _rest_note_data(duration: Fraction) -> NoteData:
    vexflow_duration, dots = _duration_to_vexflow(duration)
    return NoteData(keys=[_REST_KEY], duration=_rest_duration(vexflow_duration), dots=dots)


def _split_duration(duration: Fraction) -> list[Fraction]:
    if _is_supported_duration(duration):
        return [duration]

    remaining = duration
    fragments: list[Fraction] = []
    while remaining:
        fragment = next((candidate for candidate in _REPRESENTABLE_DURATIONS if candidate <= remaining), None)
        if fragment is None:
            raise UnsupportedNotationDurationError(
                f"duration {duration} is not supported by the VexFlow notebook bridge"
            )

        fragments.append(fragment)
        remaining -= fragment

    return fragments


def _is_supported_duration(duration: Fraction) -> bool:
    try:
        _duration_to_vexflow(duration)
    except UnsupportedNotationDurationError:
        return False

    return True


def _duration_to_vexflow(duration: Fraction) -> tuple[VexflowDuration, int]:
    for dots in range(_MAX_DOTS + 1):
        multiplier = Fraction(2 ** (dots + 1) - 1, 2**dots)
        base_duration = duration / multiplier
        if base_duration in _POWER_DURATION_BY_FRACTION:
            return _POWER_DURATION_BY_FRACTION[base_duration], dots  # type: ignore[return-value]

    raise UnsupportedNotationDurationError(f"duration {duration} is not supported by the VexFlow notebook bridge")


def _rest_duration(duration: VexflowDuration) -> VexflowDuration:
    return f"{duration}{REST_SUFFIX}"  # type: ignore[return-value]


def _max_notes_per_measure(rows: list[list[StaveData]]) -> int | None:
    counts = [len(stave.voices[0].notes) for row in rows for stave in row if stave.voices]
    if not counts:
        return None

    return max(counts)
