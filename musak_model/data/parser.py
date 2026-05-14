from fractions import Fraction
from pathlib import Path
from typing import Final

from music21 import chord, converter
from music21 import key as m21_key
from music21 import note
from music21.meter.base import TimeSignature
from music21.stream.base import Measure, Part, Score

from musak_model.common.elements import PITCHES_PER_OCTAVE
from musak_model.data.hand_selection import select_piano_hand_parts
from musak_model.data.schema import (
    ParsedBar,
    ParsedChord,
    ParsedEvent,
    ParsedNote,
    ParsedRest,
    ParsedScore,
)
from musak_model.tokens.schema import ScaleType

_QUARTER_NOTE_FRACTION: Final[Fraction] = Fraction(1, 4)
_TRIPLET_DENOMINATOR_LIMIT: Final[int] = 12

_DEFAULT_KEY_FIFTHS: Final[int] = 0
_DEFAULT_SCALE_TYPE: Final[ScaleType] = ScaleType.MAJOR


def parse_score(path: Path) -> ParsedScore:
    raw = converter.parse(str(path))
    if not isinstance(raw, Score):
        raise ValueError(f"expected a Score, got {type(raw).__name__}")

    key_root, key_fifths, scale_type = _detect_key(raw)
    time_numerator, time_denominator = _detect_time_signature(raw)
    right_hand_bars, left_hand_bars = _extract_hands(
        raw,
        default_time_signature=(time_numerator, time_denominator),
        default_key_fifths=key_fifths,
    )

    return ParsedScore(
        key_root=key_root,
        key_fifths=key_fifths,
        scale_type=scale_type,
        time_numerator=time_numerator,
        time_denominator=time_denominator,
        right_hand_bars=right_hand_bars,
        left_hand_bars=left_hand_bars,
    )


def _detect_key(score: object) -> tuple[int, int, ScaleType]:
    if not isinstance(score, Score):
        raise TypeError(f"expected Score, got {type(score).__name__}")

    key_signature = _first_score_key_signature(score)
    if key_signature is None:
        key_fifths = _DEFAULT_KEY_FIFTHS
        scale_type = _DEFAULT_SCALE_TYPE
    else:
        key_fifths = int(key_signature.sharps)
        scale_type = _key_signature_scale_type(key_signature)

    key_root = _key_root_from_fifths(key_fifths, scale_type=scale_type)
    return key_root, key_fifths, scale_type


def _first_score_key_signature(score: Score) -> m21_key.KeySignature | None:
    key_signatures = list(score.recurse().getElementsByClass(m21_key.KeySignature))
    if not key_signatures:
        return None

    return key_signatures[0]


def _key_signature_scale_type(key_signature: m21_key.KeySignature) -> ScaleType:
    mode = getattr(key_signature, "mode", None)
    if mode == "minor":
        return ScaleType.HARMONIC_MINOR

    return ScaleType.MAJOR


def _key_root_from_fifths(key_fifths: int, *, scale_type: ScaleType) -> int:
    major_tonic = (key_fifths * 7) % PITCHES_PER_OCTAVE
    if scale_type == ScaleType.MAJOR:
        return major_tonic

    if scale_type == ScaleType.HARMONIC_MINOR:
        return (major_tonic + 9) % PITCHES_PER_OCTAVE

    raise ValueError(f"unsupported key signature scale type: {scale_type.value}")


def _detect_time_signature(score: object) -> tuple[int, int]:
    if not isinstance(score, Score):
        raise TypeError(f"expected Score, got {type(score).__name__}")

    time_signatures = list(score.recurse().getElementsByClass(TimeSignature))
    if not time_signatures:
        raise ValueError("no time signature found in score")

    first = time_signatures[0]
    return int(first.numerator), int(first.denominator)


def _extract_hands(
    score: object,
    *,
    default_time_signature: tuple[int, int],
    default_key_fifths: int,
) -> tuple[list[ParsedBar], list[ParsedBar]]:
    if not isinstance(score, Score):
        raise TypeError(f"expected Score, got {type(score).__name__}")

    hand_parts = select_piano_hand_parts(score)

    right_hand_bars = _extract_bars(
        hand_parts.right,
        default_time_signature=default_time_signature,
        default_key_fifths=default_key_fifths,
    )
    left_hand_bars = _extract_bars(
        hand_parts.left,
        default_time_signature=default_time_signature,
        default_key_fifths=default_key_fifths,
    )
    return right_hand_bars, left_hand_bars


def _extract_bars(
    part: object,
    *,
    default_time_signature: tuple[int, int],
    default_key_fifths: int,
) -> list[ParsedBar]:
    if not isinstance(part, Part):
        raise TypeError(f"expected Part, got {type(part).__name__}")

    measures = list(part.getElementsByClass(Measure))
    return [
        _parse_measure(
            measure,
            default_time_signature=default_time_signature,
            default_key_fifths=default_key_fifths,
        )
        for measure in measures
    ]


def _parse_measure(
    measure: object,
    *,
    default_time_signature: tuple[int, int],
    default_key_fifths: int,
) -> ParsedBar:
    if not isinstance(measure, Measure):
        raise TypeError(f"expected Measure, got {type(measure).__name__}")

    time_numerator, time_denominator = _measure_time_signature(
        measure,
        default_time_signature=default_time_signature,
    )
    key_fifths = _measure_key_fifths(measure, default_key_fifths=default_key_fifths)
    events: list[ParsedEvent] = []
    for element in measure.flatten().notesAndRests:
        beat_offset = _to_fraction(element.offset) * _QUARTER_NOTE_FRACTION
        duration = _to_fraction(element.duration.quarterLength) * _QUARTER_NOTE_FRACTION
        if duration <= 0:
            continue

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

    return ParsedBar(
        time_numerator=time_numerator,
        time_denominator=time_denominator,
        key_fifths=key_fifths,
        events=events,
    )


def _measure_time_signature(measure: Measure, *, default_time_signature: tuple[int, int]) -> tuple[int, int]:
    local_time_signatures = list(measure.getElementsByClass(TimeSignature))
    if local_time_signatures:
        first = local_time_signatures[0]
        return int(first.numerator), int(first.denominator)

    time_signature = measure.getContextByClass(TimeSignature)
    if isinstance(time_signature, TimeSignature):
        return int(time_signature.numerator), int(time_signature.denominator)

    return default_time_signature


def _measure_key_fifths(measure: Measure, *, default_key_fifths: int) -> int:
    local_key_signatures = list(measure.getElementsByClass(m21_key.KeySignature))
    if local_key_signatures:
        return int(local_key_signatures[0].sharps)

    key_signature = measure.getContextByClass(m21_key.KeySignature)
    if isinstance(key_signature, m21_key.KeySignature):
        return int(key_signature.sharps)

    return default_key_fifths


def _to_fraction(value: float | Fraction) -> Fraction:
    return Fraction(value).limit_denominator(_TRIPLET_DENOMINATOR_LIMIT)
