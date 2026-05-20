from fractions import Fraction
from pathlib import Path
from typing import Final

from music21 import chord, converter
from music21 import key as m21_key
from music21 import note
from music21.meter.base import TimeSignature
from music21.stream.base import Measure, Part, Score

from musak_model.data.hand_selection import select_piano_hand_parts
from musak_model.data.schema import (
    ParsedBar,
    ParsedChord,
    ParsedEvent,
    ParsedNote,
    ParsedRest,
    ParsedScore,
    TieType,
)
from musak_model.tokens.schema import ScaleType
from musak_shared.elements import PITCHES_PER_OCTAVE

_QUARTER_NOTE_FRACTION: Final[Fraction] = Fraction(1, 4)
_TRIPLET_DENOMINATOR_LIMIT: Final[int] = 12

_DEFAULT_KEY_FIFTHS: Final[int] = 0
_DEFAULT_SCALE_TYPE: Final[ScaleType] = ScaleType.MAJOR


def parse_score(path: Path) -> ParsedScore:
    raw = converter.parse(str(path))
    score = _score_from_music21(raw)

    key_root, key_fifths, scale_type = _detect_key(score)
    time_numerator, time_denominator = _detect_time_signature(score)
    right_hand_bars, left_hand_bars = _extract_hands(
        score,
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


def _score_from_music21(raw: object) -> Score:
    if not isinstance(raw, Score):
        raise ValueError(f"expected a Score, got {type(raw).__name__}")

    return raw


def _detect_key(score: Score) -> tuple[int, int, ScaleType]:
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
    for key_signature in score.recurse().getElementsByClass(m21_key.KeySignature):
        return _key_signature_from_music21(key_signature)

    return None


def _key_signature_from_music21(raw: object) -> m21_key.KeySignature:
    if not isinstance(raw, m21_key.KeySignature):
        raise TypeError(f"expected KeySignature, got {type(raw).__name__}")

    return raw


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


def _detect_time_signature(score: Score) -> tuple[int, int]:
    for time_signature in score.recurse().getElementsByClass(TimeSignature):
        first = _time_signature_from_music21(time_signature)
        return int(first.numerator), int(first.denominator)

    raise ValueError("no time signature found in score")


def _time_signature_from_music21(raw: object) -> TimeSignature:
    if not isinstance(raw, TimeSignature):
        raise TypeError(f"expected TimeSignature, got {type(raw).__name__}")

    return raw


def _extract_hands(
    score: Score,
    *,
    default_time_signature: tuple[int, int],
    default_key_fifths: int,
) -> tuple[list[ParsedBar], list[ParsedBar]]:
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
    part: Part,
    *,
    default_time_signature: tuple[int, int],
    default_key_fifths: int,
) -> list[ParsedBar]:
    return [
        _parse_measure(
            _measure_from_music21(measure),
            default_time_signature=default_time_signature,
            default_key_fifths=default_key_fifths,
        )
        for measure in part.getElementsByClass(Measure)
    ]


def _measure_from_music21(raw: object) -> Measure:
    if not isinstance(raw, Measure):
        raise TypeError(f"expected Measure, got {type(raw).__name__}")

    return raw


def _parse_measure(
    measure: Measure,
    *,
    default_time_signature: tuple[int, int],
    default_key_fifths: int,
) -> ParsedBar:
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
                    tie_type=_note_tie_type(element),
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
                    tie_type=_chord_tie_type(element),
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


def _note_tie_type(element: note.Note) -> TieType | None:
    if element.tie is None:
        return None

    return _tie_type_from_text(element.tie.type)


def _chord_tie_type(element: chord.Chord) -> TieType | None:
    tie_types = [_note_tie_type(chord_note) for chord_note in element.notes]
    present_tie_types = {tie_type for tie_type in tie_types if tie_type is not None}
    if not present_tie_types:
        return None

    if len(present_tie_types) > 1 or any(tie_type is None for tie_type in tie_types):
        return TieType.PARTIAL

    return present_tie_types.pop()


def _tie_type_from_text(value: str) -> TieType:
    match value:
        case "start":
            return TieType.START
        case "continue":
            return TieType.CONTINUE
        case "stop":
            return TieType.STOP
        case _:
            raise ValueError(f"unsupported tie type: {value}")


def _to_fraction(value: float | Fraction) -> Fraction:
    return Fraction(value).limit_denominator(_TRIPLET_DENOMINATOR_LIMIT)
