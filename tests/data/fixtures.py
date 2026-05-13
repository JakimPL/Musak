from fractions import Fraction

from musak_model.data.schema import ParsedBar, ParsedChord, ParsedEvent, ParsedNote, ParsedRest, ParsedScore


def note_event(*, midi_pitch: int, duration: Fraction, beat_offset: Fraction) -> ParsedNote:
    return ParsedNote(midi_pitch=midi_pitch, duration=duration, beat_offset=beat_offset)


def rest_event(*, duration: Fraction, beat_offset: Fraction) -> ParsedRest:
    return ParsedRest(duration=duration, beat_offset=beat_offset)


def chord_event(*, midi_pitches: list[int], duration: Fraction, beat_offset: Fraction) -> ParsedChord:
    return ParsedChord(midi_pitches=midi_pitches, duration=duration, beat_offset=beat_offset)


def bar(events: list[ParsedEvent]) -> ParsedBar:
    return ParsedBar(events=events)


def parsed_score(
    *,
    right_hand_bars: list[ParsedBar],
    left_hand_bars: list[ParsedBar],
    key_root: int = 0,
    mode: str = "major",
    time_numerator: int = 4,
    time_denominator: int = 4,
) -> ParsedScore:
    return ParsedScore(
        key_root=key_root,
        mode=mode,
        time_numerator=time_numerator,
        time_denominator=time_denominator,
        right_hand_bars=right_hand_bars,
        left_hand_bars=left_hand_bars,
    )
