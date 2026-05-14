from fractions import Fraction
from pathlib import Path
from typing import Final

from music21 import chord, converter
from music21 import key as m21_key
from music21 import note
from music21.meter.base import TimeSignature
from music21.stream.base import Measure, Part, Score

from musak_model.common.elements import PITCHES_PER_OCTAVE
from musak_model.data.schema import (
    ParsedBar,
    ParsedChord,
    ParsedEvent,
    ParsedNote,
    ParsedRest,
    ParsedScore,
)

_QUARTER_NOTE_FRACTION: Final[Fraction] = Fraction(1, 4)
_TRIPLET_DENOMINATOR_LIMIT: Final[int] = 12

_TREBLE_PART_INDEX: Final[int] = 0
_BASS_PART_INDEX: Final[int] = 1


def parse_score(path: Path) -> ParsedScore:
    raw = converter.parse(str(path))
    if not isinstance(raw, Score):
        raise ValueError(f"expected a Score, got {type(raw).__name__}")

    key_root, key_fifths, mode = _detect_key(raw)
    time_numerator, time_denominator = _detect_time_signature(raw)
    right_hand_bars, left_hand_bars = _extract_hands(raw)

    return ParsedScore(
        key_root=key_root,
        key_fifths=key_fifths,
        mode=mode,
        time_numerator=time_numerator,
        time_denominator=time_denominator,
        right_hand_bars=right_hand_bars,
        left_hand_bars=left_hand_bars,
    )


def _detect_key(score: object) -> tuple[int, int, str]:
    if not isinstance(score, Score):
        raise TypeError(f"expected Score, got {type(score).__name__}")

    key_signature = score.analyze("key")
    if not isinstance(key_signature, m21_key.Key):
        raise ValueError(f"unexpected key analysis result: {type(key_signature)}")

    tonic = key_signature.tonic
    if tonic is None:
        raise ValueError("key signature has no tonic")

    return tonic.midi % PITCHES_PER_OCTAVE, key_signature.sharps, key_signature.mode


def _detect_time_signature(score: object) -> tuple[int, int]:
    if not isinstance(score, Score):
        raise TypeError(f"expected Score, got {type(score).__name__}")

    time_signatures = list(score.recurse().getElementsByClass(TimeSignature))
    if not time_signatures:
        raise ValueError("no time signature found in score")

    first = time_signatures[0]
    return int(first.numerator), int(first.denominator)


def _extract_hands(score: object) -> tuple[list[ParsedBar], list[ParsedBar]]:
    if not isinstance(score, Score):
        raise TypeError(f"expected Score, got {type(score).__name__}")

    parts = [part for part in score.parts if isinstance(part, Part)]
    if len(parts) < 2:
        raise ValueError(f"expected at least 2 parts (hands), found {len(parts)}")

    right_hand_bars = _extract_bars(parts[_TREBLE_PART_INDEX])
    left_hand_bars = _extract_bars(parts[_BASS_PART_INDEX])
    return right_hand_bars, left_hand_bars


def _extract_bars(part: object) -> list[ParsedBar]:
    if not isinstance(part, Part):
        raise TypeError(f"expected Part, got {type(part).__name__}")

    measures = list(part.getElementsByClass(Measure))
    return [_parse_measure(measure) for measure in measures]


def _parse_measure(measure: object) -> ParsedBar:
    if not isinstance(measure, Measure):
        raise TypeError(f"expected Measure, got {type(measure).__name__}")

    events: list[ParsedEvent] = []
    for element in measure.flatten().notesAndRests:
        beat_offset = _to_fraction(element.offset) * _QUARTER_NOTE_FRACTION
        duration = _to_fraction(element.duration.quarterLength) * _QUARTER_NOTE_FRACTION

        if isinstance(element, note.Note):
            events.append(
                ParsedNote(
                    midi_pitch=element.pitch.midi,
                    duration=duration,
                    beat_offset=beat_offset,
                )
            )

        elif isinstance(element, note.Rest):
            events.append(ParsedRest(duration=duration, beat_offset=beat_offset))

        elif isinstance(element, chord.Chord):
            midi_pitches = [pitch.midi for pitch in element.pitches]
            events.append(
                ParsedChord(
                    midi_pitches=midi_pitches,
                    duration=duration,
                    beat_offset=beat_offset,
                )
            )

    return ParsedBar(events=events)


def _to_fraction(value: float | Fraction) -> Fraction:
    return Fraction(value).limit_denominator(_TRIPLET_DENOMINATOR_LIMIT)
