from fractions import Fraction
from pathlib import Path

from musak_model.data.schema import Segment, SegmentMetadata
from musak_model.harmony.decoding.config import ChordDecoderConfig
from musak_model.harmony.decoding.decoder import ViterbiChordDecoder
from musak_model.harmony.schema import Chord, ChordQuality
from musak_model.harmony.vocabulary import ChordVocabularyConfig
from musak_model.n_grams.profile.chord.extraction import chord_statistics
from musak_model.n_grams.profile.chord.schema import (
    INITIAL_CHORD_SOURCE,
    ChordDecodeSpec,
    ChordTransitionKey,
    chord_to_key,
)
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import BarToken, Hand, HandToken, JoinWithPreviousToken, NoteToken, ScaleType, Token

_TONIC = chord_to_key(Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR))
_SUBDOMINANT = chord_to_key(Chord(root_degree=4, root_accidental=0, quality=ChordQuality.MAJOR))
_DOMINANT = chord_to_key(Chord(root_degree=5, root_accidental=0, quality=ChordQuality.MAJOR))


def _triad(notes: list[tuple[int, int]], *, duration_id: int) -> list[Token]:
    tokens: list[Token] = []
    for index, (degree, octave_offset) in enumerate(notes):
        tokens.append(NoteToken(degree=degree, accidental=0, octave_offset=octave_offset, duration_id=duration_id))
        if index > 0:
            tokens.append(JoinWithPreviousToken())

    return tokens


def _progression_segment(duration_vocabulary: DurationVocabulary) -> Segment:
    whole = duration_vocabulary.require_duration_id(Fraction(1))
    bars = [
        _triad([(1, 0), (3, 0), (5, 0)], duration_id=whole),
        _triad([(4, 0), (6, 0), (1, 1)], duration_id=whole),
        _triad([(5, 0), (7, 0), (2, 0)], duration_id=whole),
        _triad([(1, 0), (3, 0), (5, 0)], duration_id=whole),
    ]
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


def _melodic_segment(duration_vocabulary: DurationVocabulary) -> Segment:
    quarter = duration_vocabulary.require_duration_id(Fraction(1, 4))
    bars = [
        [(1, 0), (3, 0), (5, 0), (3, 0)],
        [(5, 0), (7, 0), (2, 1), (7, 0)],
    ]
    tokens: list[Token] = [HandToken(hand=Hand.RIGHT)]
    for bar_notes in bars:
        for degree, octave_offset in bar_notes:
            tokens.append(NoteToken(degree=degree, accidental=0, octave_offset=octave_offset, duration_id=quarter))
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
            source_file=Path("melody.mxl"),
            difficulty_level=None,
        ),
    )


def _decode_spec() -> ChordDecodeSpec:
    return ChordDecodeSpec(
        decoder_config=ChordDecoderConfig(resolution=1, self_transition_bias=0.25, non_chord_penalty=1.0),
        vocabulary=ChordVocabularyConfig.load(),
    )


def test_chord_statistics_counts_adjacent_transitions_with_initial(
    duration_vocabulary: DurationVocabulary,
) -> None:
    statistics = chord_statistics(
        [_progression_segment(duration_vocabulary)],
        duration_vocabulary=duration_vocabulary,
        decode_spec=_decode_spec(),
        min_n=2,
        max_n=2,
    )

    assert statistics.transition_counts == {
        ChordTransitionKey(INITIAL_CHORD_SOURCE, _TONIC): 1,
        ChordTransitionKey(_TONIC, _SUBDOMINANT): 1,
        ChordTransitionKey(_SUBDOMINANT, _DOMINANT): 1,
        ChordTransitionKey(_DOMINANT, _TONIC): 1,
    }


def test_chord_statistics_assigns_each_figure_to_its_onset_chord(
    duration_vocabulary: DurationVocabulary,
) -> None:
    segment = _melodic_segment(duration_vocabulary)
    spec = _decode_spec()
    windows = ViterbiChordDecoder(config=spec.decoder_config).decode(
        segment, duration_vocabulary=duration_vocabulary, vocabulary=spec.vocabulary
    )

    statistics = chord_statistics(
        [segment], duration_vocabulary=duration_vocabulary, decode_spec=spec, min_n=2, max_n=2
    )

    chords_used = {key.chord for key in statistics.figure_by_chord_counts}
    # Each four-onset bar yields three length-2 figures, all covered by that bar's single decoded window.
    assert sum(statistics.figure_by_chord_counts.values()) == 6
    assert chords_used == {chord_to_key(window.chord) for window in windows}
