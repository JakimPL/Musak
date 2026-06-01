from fractions import Fraction
from pathlib import Path

from musak_model.data.schema import Segment, SegmentMetadata
from musak_model.harmony.decoding.config import ChordDecoderConfig
from musak_model.harmony.decoding.decoder import ViterbiChordDecoder
from musak_model.harmony.vocabulary import ChordVocabularyConfig
from musak_model.synthetic.fitting.form.analysis import analyze_segment
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import BarToken, Hand, HandToken, JoinWithPreviousToken, NoteToken, ScaleType, Token
from musak_shared.elements import HarmonicFunction

_NoteSpec = tuple[int, int, int]


def _triad(notes: list[_NoteSpec], *, duration_id: int) -> list[Token]:
    tokens: list[Token] = []
    for index, (degree, accidental, octave_offset) in enumerate(notes):
        tokens.append(
            NoteToken(degree=degree, accidental=accidental, octave_offset=octave_offset, duration_id=duration_id)
        )
        if index > 0:
            tokens.append(JoinWithPreviousToken())

    return tokens


def _segment(bars: list[list[Token]]) -> Segment:
    tokens: list[Token] = [HandToken(hand=Hand.RIGHT)]
    for bar_tokens in bars:
        tokens.extend(bar_tokens)
        tokens.append(BarToken())

    return Segment(
        tokens=tokens,
        metadata=SegmentMetadata(
            scale_root=0,
            scale_type=ScaleType.MAJOR,
            time_numerator=4,
            time_denominator=4,
            bar_count=len(bars),
            window_start_bar=0,
            source_file=Path("sample.mxl"),
            difficulty_level=None,
        ),
    )


def _decoder() -> ViterbiChordDecoder:
    return ViterbiChordDecoder(
        config=ChordDecoderConfig(resolution=1, self_transition_bias=0.25, non_chord_penalty=1.0)
    )


def test_analyze_segment_maps_progression_to_functions(duration_vocabulary: DurationVocabulary) -> None:
    whole = duration_vocabulary.require_duration_id(Fraction(1))
    segment = _segment(
        [
            _triad([(1, 0, 0), (3, 0, 0), (5, 0, 0)], duration_id=whole),
            _triad([(4, 0, 0), (6, 0, 0), (1, 0, 1)], duration_id=whole),
            _triad([(5, 0, 0), (7, 0, 0), (2, 0, 0)], duration_id=whole),
            _triad([(1, 0, 0), (3, 0, 0), (5, 0, 0)], duration_id=whole),
        ]
    )

    piece = analyze_segment(
        segment,
        chord_decoder=_decoder(),
        chord_vocabulary=ChordVocabularyConfig.load(),
        duration_vocabulary=duration_vocabulary,
        figure_min_n=1,
        figure_max_n=4,
    )

    assert piece is not None
    assert [slot.function for slot in piece.slots] == [
        HarmonicFunction.TONIC,
        HarmonicFunction.PREDOMINANT,
        HarmonicFunction.DOMINANT,
        HarmonicFunction.TONIC,
    ]
    assert piece.slots[0].tonic_triad_overlap == 1.0
    assert all(0.0 < slot.metrical_weight <= 1.0 for slot in piece.slots)
    assert all(slot.dwell == 1.0 for slot in piece.slots)
    assert len(piece.bar_figures) == 4


def test_analyze_segment_returns_none_for_empty_token_stream(duration_vocabulary: DurationVocabulary) -> None:
    segment = _segment([])

    piece = analyze_segment(
        segment,
        chord_decoder=_decoder(),
        chord_vocabulary=ChordVocabularyConfig.load(),
        duration_vocabulary=duration_vocabulary,
        figure_min_n=1,
        figure_max_n=4,
    )

    assert piece is None
