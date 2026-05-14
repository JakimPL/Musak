from collections.abc import Iterable
from enum import StrEnum
from fractions import Fraction
from typing import Final

import pandas as pd

from musak_model.data.schema import ParsedScore, Segment
from musak_model.decoder import PianoRollEvent, parsed_score_to_piano_roll_events, segment_to_piano_roll_events
from musak_model.tokens.duration import DurationVocabulary

_QUARTERS_PER_WHOLE: Final[int] = 4
_SECONDS_PER_MINUTE: Final[int] = 60
_SHARP_PITCH_NAMES: Final[tuple[str, ...]] = ("C-", "C#", "D-", "D#", "E-", "F-", "F#", "G-", "G#", "A-", "A#", "B-")
_FLAT_PITCH_NAMES: Final[tuple[str, ...]] = ("C-", "Db", "D-", "Eb", "E-", "F-", "Gb", "G-", "Ab", "A-", "Bb", "B-")


class PitchSpelling(StrEnum):
    SHARPS = "sharps"
    FLATS = "flats"


def piano_roll_dataframe(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
    pitch_spelling: PitchSpelling = PitchSpelling.SHARPS,
    bpm: int = 60,
) -> pd.DataFrame:
    return _events_to_dataframe(
        segment_to_piano_roll_events(segment, duration_vocabulary=duration_vocabulary),
        measure_duration=Fraction(segment.time_numerator, segment.time_denominator),
        window_start_bar=segment.metadata.window_start_bar,
        pitch_spelling=pitch_spelling,
        bpm=bpm,
    )


def parsed_score_piano_roll_dataframe(
    score: ParsedScore,
    *,
    pitch_spelling: PitchSpelling = PitchSpelling.SHARPS,
    bpm: int = 60,
) -> pd.DataFrame:
    return _events_to_dataframe(
        parsed_score_to_piano_roll_events(score),
        measure_duration=Fraction(score.time_numerator, score.time_denominator),
        window_start_bar=0,
        pitch_spelling=pitch_spelling,
        bpm=bpm,
    )


def midi_pitch_name(midi_pitch: int, *, pitch_spelling: PitchSpelling = PitchSpelling.SHARPS) -> str:
    pitch_names = _FLAT_PITCH_NAMES if pitch_spelling == PitchSpelling.FLATS else _SHARP_PITCH_NAMES
    pitch_class = midi_pitch % len(pitch_names)
    octave = midi_pitch // len(pitch_names) - 1
    return f"{pitch_names[pitch_class]}{octave}"


def _events_to_dataframe(
    events: Iterable[PianoRollEvent],
    *,
    measure_duration: Fraction,
    window_start_bar: int,
    pitch_spelling: PitchSpelling,
    bpm: int,
) -> pd.DataFrame:
    rows = []
    for event in events:
        start_seconds = _whole_note_fraction_to_seconds(event.start, bpm=bpm)
        duration_seconds = _whole_note_fraction_to_seconds(event.duration, bpm=bpm)
        bar_start = window_start_bar + 1 + event.start / measure_duration
        bar_duration = event.duration / measure_duration
        duration_text = _fraction_text(event.duration)
        rows.append(
            {
                "hand": event.hand.value,
                "midi_pitch": event.midi_pitch,
                "pitch": midi_pitch_name(event.midi_pitch, pitch_spelling=pitch_spelling),
                "start": float(event.start),
                "duration": float(event.duration),
                "duration_fraction": duration_text,
                "end": float(event.end),
                "bar_start": float(bar_start),
                "bar_end": float(bar_start + bar_duration),
                "start_seconds": start_seconds,
                "duration_seconds": duration_seconds,
                "end_seconds": start_seconds + duration_seconds,
                "token_index": event.token_index,
                "token": event.token_text,
            }
        )

    return pd.DataFrame(rows)


def _whole_note_fraction_to_seconds(duration: Fraction, *, bpm: int) -> float:
    return float(duration * _QUARTERS_PER_WHOLE * _SECONDS_PER_MINUTE / bpm)


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}:{value.denominator}"
