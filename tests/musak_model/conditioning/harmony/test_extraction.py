from fractions import Fraction
from pathlib import Path

from musak_model.conditioning.harmony.extraction import (
    harmonic_plan_windows_from_segment,
)
from musak_model.data.schema import Segment, SegmentMetadata
from musak_model.data.tokenization_context import tokenization_context_from_scale
from musak_model.harmony.decoding import ChordDecoderConfig, ViterbiChordDecoder
from musak_model.harmony.schema import ChordQuality
from musak_model.harmony.vocabulary import ChordVocabularyConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import (
    BarToken,
    JoinWithPreviousToken,
    NoteToken,
    ScaleType,
)


def test_harmonic_plan_windows_from_segment_reuses_viterbi_decoder(
    duration_vocabulary: DurationVocabulary,
) -> None:
    whole_id = duration_vocabulary.require_duration_id(Fraction(1))
    segment = Segment(
        tokens=[
            NoteToken(degree=5, accidental=0, octave_offset=0, duration_id=whole_id),
            NoteToken(degree=7, accidental=0, octave_offset=0, duration_id=whole_id),
            JoinWithPreviousToken(),
            NoteToken(degree=2, accidental=0, octave_offset=0, duration_id=whole_id),
            JoinWithPreviousToken(),
            BarToken(),
        ],
        metadata=SegmentMetadata(
            scale_root=0,
            scale_type=ScaleType.MAJOR,
            tokenization_context=tokenization_context_from_scale(scale_root=0, scale_type=ScaleType.MAJOR),
            time_numerator=4,
            time_denominator=4,
            bar_count=1,
            window_start_bar=0,
            source_file=Path("sample.mxl"),
            difficulty_level=None,
        ),
    )

    windows = harmonic_plan_windows_from_segment(
        segment,
        decoder=ViterbiChordDecoder(
            config=ChordDecoderConfig(
                resolution=1,
                self_transition_bias=0.25,
                non_chord_penalty=1.0,
            )
        ),
        duration_vocabulary=duration_vocabulary,
        vocabulary=ChordVocabularyConfig.load(),
    )

    assert len(windows) == 1
    assert windows[0].start == Fraction(0)
    assert windows[0].end == Fraction(1)
    assert (windows[0].chord.root_degree, windows[0].chord.quality) == (5, ChordQuality.MAJOR)
