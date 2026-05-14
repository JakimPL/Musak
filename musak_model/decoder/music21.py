from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from music21 import chord, note, stream
from music21.base import Music21Object
from music21.meter.base import TimeSignature

from musak_model.common.elements import QUARTER_NOTE_DURATION
from musak_model.data.schema import Segment
from musak_model.decoder.piano_roll import PianoRollEvent, segment_to_piano_roll_events
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand


def segment_to_music21_score(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
) -> stream.Score:
    events = segment_to_piano_roll_events(segment, duration_vocabulary=duration_vocabulary)
    score = stream.Score()
    score.insert(0, _part_from_events(events, hand=Hand.RIGHT, segment=segment))  # type: ignore[no-untyped-call]
    score.insert(0, _part_from_events(events, hand=Hand.LEFT, segment=segment))  # type: ignore[no-untyped-call]
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


def _part_from_events(events: list[PianoRollEvent], *, hand: Hand, segment: Segment) -> stream.Part:
    part = stream.Part(id=hand.value)  # type: ignore[no-untyped-call]
    part.insert(0, TimeSignature(f"{segment.time_numerator}/{segment.time_denominator}"))  # type: ignore[no-untyped-call]
    hand_events = [event for event in events if event.hand == hand]
    grouped = _group_events_by_start(hand_events)

    for start, onset_events in sorted(grouped.items(), key=lambda item: item[0]):
        offset = _fraction_to_quarter_length(start)
        duration = _fraction_to_quarter_length(max(event.duration for event in onset_events))
        if len(onset_events) == 1:
            element: Music21Object = note.Note(onset_events[0].midi_pitch)
        else:
            element = chord.Chord([event.midi_pitch for event in onset_events])

        element.duration.quarterLength = duration
        part.insert(offset, element)  # type: ignore[no-untyped-call]

    return part


def _group_events_by_start(events: list[PianoRollEvent]) -> dict[Fraction, list[PianoRollEvent]]:
    grouped: dict[Fraction, list[PianoRollEvent]] = defaultdict(list)
    for event in events:
        grouped[event.start].append(event)

    return dict(grouped)


def _fraction_to_quarter_length(value: Fraction) -> Fraction:
    return value / QUARTER_NOTE_DURATION
