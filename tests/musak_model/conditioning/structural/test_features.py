from fractions import Fraction
from pathlib import Path

from musak_model.conditioning.structural.features import extract_structural_control_features
from musak_model.data.schema import Segment, SegmentMetadata
from musak_model.data.tokenization_context import tokenization_context_from_scale
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand, HandToken, JoinWithPreviousToken, NoteToken, RestToken, ScaleType


def _metadata() -> SegmentMetadata:
    return SegmentMetadata(
        scale_root=0,
        scale_type=ScaleType.MAJOR,
        tokenization_context=tokenization_context_from_scale(scale_root=0, scale_type=ScaleType.MAJOR),
        time_numerator=4,
        time_denominator=4,
        bar_count=1,
        window_start_bar=0,
        source_file=Path("piece.mxl"),
        difficulty_level=None,
    )


def test_extract_structural_control_features_from_tokens(duration_vocabulary: DurationVocabulary) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    eighth_id = duration_vocabulary.fraction_to_id(Fraction(1, 8))
    dotted_quarter_id = duration_vocabulary.fraction_to_id(Fraction(3, 8))
    segment = Segment(
        tokens=[
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
            NoteToken(degree=3, accidental=0, octave_offset=0, duration_id=quarter_id),
            JoinWithPreviousToken(),
            NoteToken(degree=5, accidental=0, octave_offset=0, duration_id=eighth_id),
            HandToken(hand=Hand.LEFT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=dotted_quarter_id),
            RestToken(duration_id=eighth_id),
        ],
        metadata=_metadata(),
    )

    features = extract_structural_control_features(segment, duration_vocabulary=duration_vocabulary)

    assert features.shortest_note_duration == Fraction(1, 8)
    assert features.has_dotted_notes is True
    assert features.max_notes_per_onset == 2
    assert features.max_notes_per_hand == 2
    assert features.max_onset_span_semitones == 4
    assert features.max_melodic_gap_semitones == 3
    assert features.static_hand_span_degrees == 5
    assert features.bar_count is None


def test_extract_structural_control_features_separates_total_onset_and_per_hand_counts(
    duration_vocabulary: DurationVocabulary,
) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    segment = Segment(
        tokens=[
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
            NoteToken(degree=3, accidental=0, octave_offset=0, duration_id=quarter_id),
            JoinWithPreviousToken(),
            HandToken(hand=Hand.LEFT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
            JoinWithPreviousToken(),
            NoteToken(degree=3, accidental=0, octave_offset=0, duration_id=quarter_id),
            JoinWithPreviousToken(),
        ],
        metadata=_metadata(),
    )

    features = extract_structural_control_features(segment, duration_vocabulary=duration_vocabulary)

    assert features.max_notes_per_onset == 4
    assert features.max_notes_per_hand == 2
