from fractions import Fraction
from pathlib import Path

from musak_model.data.config import SegmentationConfig
from musak_model.data.pipeline import segment_parsed_score
from musak_model.data.schema import ParsedBar, ParsedNote, ParsedScore, SegmentIneligibilityReason
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import ScaleType


def _empty_bar() -> ParsedBar:
    return ParsedBar(time_numerator=4, time_denominator=4, key_fifths=0, events=[])


def test_segment_parsed_score_keeps_recoverable_segments_when_feature_extraction_fails(
    duration_vocabulary: DurationVocabulary,
) -> None:
    bad_register_bar = ParsedBar(
        time_numerator=4,
        time_denominator=4,
        key_fifths=0,
        events=[ParsedNote(midi_pitch=24, duration=Fraction(1, 4), beat_offset=Fraction(0))],
    )
    score = ParsedScore(
        key_root=0,
        key_fifths=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        right_hand_bars=[_empty_bar(), bad_register_bar, _empty_bar()],
        left_hand_bars=[_empty_bar(), _empty_bar(), _empty_bar()],
    )

    segments = segment_parsed_score(
        score,
        Path("piece.mxl"),
        segmentation=SegmentationConfig(window_bars=1, stride_bars=1),
        duration_vocabulary=duration_vocabulary,
    )

    assert [segment.metadata.eligible_for_training for segment in segments] == [True, False, True]
    assert segments[1].metadata.ineligibility_reasons == {SegmentIneligibilityReason.TOKENIZATION_ERROR}
    assert segments[0].metadata.difficulty_features is not None
    assert segments[1].metadata.difficulty_features is None
