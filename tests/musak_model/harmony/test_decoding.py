from fractions import Fraction
from pathlib import Path

import pytest

from musak_model.data.schema import Segment, SegmentMetadata
from musak_model.data.tokenization_context import tokenization_context_from_scale
from musak_model.harmony.decoding import (
    ChordDecoderConfig,
    ViterbiChordDecoder,
)
from musak_model.harmony.schema import ChordQuality
from musak_model.harmony.vocabulary import ChordVocabularyConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import (
    BarToken,
    Hand,
    HandToken,
    JoinWithPreviousToken,
    NoteToken,
    RestToken,
    ScaleType,
    Token,
)

type NoteSpec = tuple[int, int, int]


@pytest.fixture
def vocabulary() -> ChordVocabularyConfig:
    return ChordVocabularyConfig.load()


_WHOLE_NOTE_RESOLUTION = 1
_HALF_NOTE_RESOLUTION = 2
_QUARTER_NOTE_RESOLUTION = 4


def _decoder(resolution: int) -> ViterbiChordDecoder:
    return ViterbiChordDecoder(
        config=ChordDecoderConfig(
            resolution=resolution,
            self_transition_bias=0.25,
            non_chord_penalty=1.0,
        )
    )


def _triad(notes: list[NoteSpec], *, duration_id: int) -> list[Token]:
    tokens: list[Token] = []
    for index, (degree, accidental, octave_offset) in enumerate(notes):
        tokens.append(
            NoteToken(degree=degree, accidental=accidental, octave_offset=octave_offset, duration_id=duration_id)
        )
        if index > 0:
            tokens.append(JoinWithPreviousToken())

    return tokens


def _melody(notes: list[NoteSpec], *, duration_id: int) -> list[Token]:
    return [
        NoteToken(degree=degree, accidental=accidental, octave_offset=octave_offset, duration_id=duration_id)
        for degree, accidental, octave_offset in notes
    ]


def _segment(
    bars: list[list[Token]],
    *,
    scale_type: ScaleType,
    time_numerator: int = 4,
    time_denominator: int = 4,
) -> Segment:
    tokens: list[Token] = [HandToken(hand=Hand.RIGHT)]
    for bar_tokens in bars:
        tokens.extend(bar_tokens)
        tokens.append(BarToken())

    return Segment(
        tokens=tokens,
        metadata=SegmentMetadata(
            scale_root=0,
            scale_type=scale_type,
            tokenization_context=tokenization_context_from_scale(scale_root=0, scale_type=scale_type),
            time_numerator=time_numerator,
            time_denominator=time_denominator,
            bar_count=len(bars),
            window_start_bar=0,
            source_file=Path("sample.mxl"),
            difficulty_level=None,
        ),
    )


def test_decodes_block_triad_progression(
    duration_vocabulary: DurationVocabulary,
    vocabulary: ChordVocabularyConfig,
) -> None:
    whole = duration_vocabulary.require_duration_id(Fraction(1))
    segment = _segment(
        [
            _triad([(1, 0, 0), (3, 0, 0), (5, 0, 0)], duration_id=whole),
            _triad([(4, 0, 0), (6, 0, 0), (1, 0, 1)], duration_id=whole),
            _triad([(5, 0, 0), (7, 0, 0), (2, 0, 0)], duration_id=whole),
            _triad([(1, 0, 0), (3, 0, 0), (5, 0, 0)], duration_id=whole),
        ],
        scale_type=ScaleType.MAJOR,
    )

    track = _decoder(_WHOLE_NOTE_RESOLUTION).decode(
        segment, duration_vocabulary=duration_vocabulary, vocabulary=vocabulary
    )

    assert [(window.chord.root_degree, window.chord.quality) for window in track] == [
        (1, ChordQuality.MAJOR),
        (4, ChordQuality.MAJOR),
        (5, ChordQuality.MAJOR),
        (1, ChordQuality.MAJOR),
    ]


def test_tolerates_passing_tone(
    duration_vocabulary: DurationVocabulary,
    vocabulary: ChordVocabularyConfig,
) -> None:
    quarter = duration_vocabulary.require_duration_id(Fraction(1, 4))
    segment = _segment(
        [_melody([(1, 0, 0), (2, 0, 0), (3, 0, 0), (5, 0, 0)], duration_id=quarter)],
        scale_type=ScaleType.MAJOR,
    )

    track = _decoder(_WHOLE_NOTE_RESOLUTION).decode(
        segment, duration_vocabulary=duration_vocabulary, vocabulary=vocabulary
    )

    assert len(track) == 1
    assert (track[0].chord.root_degree, track[0].chord.quality) == (1, ChordQuality.MAJOR)


def test_detects_borrowed_minor_iv(
    duration_vocabulary: DurationVocabulary,
    vocabulary: ChordVocabularyConfig,
) -> None:
    whole = duration_vocabulary.require_duration_id(Fraction(1))
    segment = _segment(
        [_triad([(4, 0, 0), (6, -1, 0), (1, 0, 1)], duration_id=whole)],
        scale_type=ScaleType.MAJOR,
    )

    track = _decoder(_WHOLE_NOTE_RESOLUTION).decode(
        segment, duration_vocabulary=duration_vocabulary, vocabulary=vocabulary
    )

    assert (track[0].chord.root_degree, track[0].chord.quality) == (4, ChordQuality.MINOR)


def test_empty_window_inherits_previous_chord(
    duration_vocabulary: DurationVocabulary,
    vocabulary: ChordVocabularyConfig,
) -> None:
    whole = duration_vocabulary.require_duration_id(Fraction(1))
    segment = _segment(
        [
            _triad([(1, 0, 0), (3, 0, 0), (5, 0, 0)], duration_id=whole),
            [RestToken(duration_id=whole)],
        ],
        scale_type=ScaleType.MAJOR,
    )

    track = _decoder(_WHOLE_NOTE_RESOLUTION).decode(
        segment, duration_vocabulary=duration_vocabulary, vocabulary=vocabulary
    )

    assert len(track) == 2
    assert track[1].chord == track[0].chord


def test_beat_resolution_produces_one_window_per_beat(
    duration_vocabulary: DurationVocabulary,
    vocabulary: ChordVocabularyConfig,
) -> None:
    quarter = duration_vocabulary.require_duration_id(Fraction(1, 4))
    segment = _segment(
        [_melody([(1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)], duration_id=quarter)],
        scale_type=ScaleType.MAJOR,
    )

    track = _decoder(_QUARTER_NOTE_RESOLUTION).decode(
        segment, duration_vocabulary=duration_vocabulary, vocabulary=vocabulary
    )

    assert len(track) == 4
    assert [window.start for window in track] == [Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)]


def test_odd_meter_truncates_chord_windows_at_barlines(
    duration_vocabulary: DurationVocabulary,
    vocabulary: ChordVocabularyConfig,
) -> None:
    quarter = duration_vocabulary.require_duration_id(Fraction(1, 4))
    segment = _segment(
        [_melody([(1, 0, 0), (3, 0, 0), (5, 0, 0)], duration_id=quarter)],
        scale_type=ScaleType.MAJOR,
        time_numerator=3,
        time_denominator=4,
    )

    track = _decoder(_HALF_NOTE_RESOLUTION).decode(
        segment, duration_vocabulary=duration_vocabulary, vocabulary=vocabulary
    )

    assert [(window.start, window.end) for window in track] == [
        (Fraction(0), Fraction(1, 2)),
        (Fraction(1, 2), Fraction(3, 4)),
    ]
    assert all(window.end > window.start for window in track)
