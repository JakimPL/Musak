from fractions import Fraction

from musak_model.data.schema import ParsedChord, ParsedEvent, ParsedNote, ParsedRest


def event_sort_key(event: ParsedEvent) -> tuple[Fraction, int]:
    return event.beat_offset, lowest_pitch(event)


def lowest_pitch(event: ParsedEvent) -> int:
    match event:
        case ParsedNote():
            return event.midi_pitch
        case ParsedChord():
            return min(event.midi_pitches)
        case ParsedRest():
            return -1


def event_pitches(event: ParsedEvent) -> tuple[int, ...]:
    match event:
        case ParsedNote():
            return (event.midi_pitch,)
        case ParsedChord():
            return tuple(sorted(event.midi_pitches))
        case ParsedRest():
            return ()
