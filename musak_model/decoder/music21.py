from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from music21 import chord, note, stream, tie
from music21.base import Music21Object
from music21.key import KeySignature
from music21.meter.base import TimeSignature

from musak_model.data.schema import Segment
from musak_model.decoder.notation import DecodedNotationEvent, segment_spelling_key_fifths, segment_to_notation_events
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand
from musak_shared.elements import QUARTER_NOTE_DURATION
from musak_shared.ratios import format_ratio


def segment_to_music21_score(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
) -> stream.Score:
    events = segment_to_notation_events(segment, duration_vocabulary=duration_vocabulary)
    score = stream.Score()
    score.insert(0, _part_from_events(events, hand=Hand.RIGHT, segment=segment))
    score.insert(0, _part_from_events(events, hand=Hand.LEFT, segment=segment))
    return score


def write_segment(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
    path: Path,
    format_name: str,
) -> Path:
    score = segment_to_music21_score(segment, duration_vocabulary=duration_vocabulary)
    written = score.write(format_name, fp=path)  # type: ignore[no-untyped-call]
    return Path(written)


def _part_from_events(events: list[DecodedNotationEvent], *, hand: Hand, segment: Segment) -> stream.Part:
    part = stream.Part(id=hand.value)  # type: ignore[no-untyped-call]
    time_signature_text = format_ratio((segment.time_numerator, segment.time_denominator))
    part.insert(0, KeySignature(segment_spelling_key_fifths(segment)))
    part.insert(0, TimeSignature(time_signature_text))
    measure_duration = Fraction(segment.time_numerator, segment.time_denominator)
    hand_events = [event for event in events if event.hand == hand]
    grouped = _group_events_by_start(hand_events)

    for start, onset_events in sorted(grouped.items(), key=lambda item: item[0]):
        onset_duration = max(event.duration for event in onset_events)
        for fragment_start, fragment_duration, tie_type in _split_at_barlines(
            start=start,
            duration=onset_duration,
            measure_duration=measure_duration,
        ):
            element = _element_from_onset_events(onset_events)
            element.duration.quarterLength = _fraction_to_quarter_length(fragment_duration)
            _apply_tie(element, tie_type)
            part.insert(_fraction_to_quarter_length(fragment_start), element)

    return part


def _group_events_by_start(events: list[DecodedNotationEvent]) -> dict[Fraction, list[DecodedNotationEvent]]:
    grouped: dict[Fraction, list[DecodedNotationEvent]] = defaultdict(list)
    for event in events:
        grouped[event.start].append(event)

    return dict(grouped)


def _split_at_barlines(
    *,
    start: Fraction,
    duration: Fraction,
    measure_duration: Fraction,
) -> list[tuple[Fraction, Fraction, str | None]]:
    if measure_duration <= 0:
        raise ValueError("measure_duration must be positive")

    end = start + duration
    cursor = start
    fragments: list[tuple[Fraction, Fraction]] = []

    while cursor < end:
        next_barline = ((cursor // measure_duration) + 1) * measure_duration
        fragment_end = min(end, next_barline)
        fragments.append((cursor, fragment_end - cursor))
        cursor = fragment_end

    if len(fragments) == 1:
        fragment_start, fragment_duration = fragments[0]
        return [(fragment_start, fragment_duration, None)]

    tied_fragments: list[tuple[Fraction, Fraction, str | None]] = []
    last_index = len(fragments) - 1
    for index, (fragment_start, fragment_duration) in enumerate(fragments):
        if index == 0:
            tie_type = "start"
        elif index == last_index:
            tie_type = "stop"
        else:
            tie_type = "continue"

        tied_fragments.append((fragment_start, fragment_duration, tie_type))

    return tied_fragments


def _element_from_onset_events(onset_events: list[DecodedNotationEvent]) -> Music21Object:
    if len(onset_events) == 1:
        return note.Note(_music21_pitch_name(onset_events[0].vexflow_key))

    return chord.Chord([_music21_pitch_name(event.vexflow_key) for event in onset_events])


def _music21_pitch_name(vexflow_key: str) -> str:
    pitch_name, octave = vexflow_key.split("/")
    letter = pitch_name[0].upper()
    accidental = pitch_name[1:].replace("b", "-")
    return f"{letter}{accidental}{octave}"


def _apply_tie(element: Music21Object, tie_type: str | None) -> None:
    if tie_type is None:
        return

    if isinstance(element, note.Note):
        element.tie = tie.Tie(tie_type)  # type: ignore[no-untyped-call]
        return

    if isinstance(element, chord.Chord):
        for chord_note in element.notes:
            chord_note.tie = tie.Tie(tie_type)  # type: ignore[no-untyped-call]


def _fraction_to_quarter_length(value: Fraction) -> Fraction:
    return value / QUARTER_NOTE_DURATION
