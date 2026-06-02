from fractions import Fraction
from pathlib import Path

from musak_model.data.schema import Segment, SegmentMetadata
from musak_model.data.tokenization_context import tokenization_context_from_scale
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import (
    BarToken,
    EndToken,
    Hand,
    HandToken,
    JoinWithPreviousToken,
    NoteToken,
    ScaleType,
)
from notebooks.utils.harmony import harmonic_plan_inspection


def test_harmonic_plan_inspection_decodes_chord_windows_and_note_diagnostics(
    duration_vocabulary: DurationVocabulary,
) -> None:
    segment = _segment(duration_vocabulary)

    inspection = harmonic_plan_inspection(segment, duration_vocabulary=duration_vocabulary)

    assert [highlight.label for highlight in inspection.chord_highlights] == ["I"]
    assert inspection.chord_highlights[0].pitch_classes == frozenset({0, 4, 7})
    assert inspection.chord_highlights[0].start_in_bars == 3.0
    assert inspection.chord_highlights[0].end_in_bars == 4.0
    assert inspection.window_frame.iloc[0]["label"] == "I"
    assert inspection.window_frame.iloc[0]["function"] == "tonic"
    assert inspection.window_frame.iloc[0]["chord_pitch_classes"] == "C E G"
    assert inspection.window_frame.iloc[0]["chord_tone_coverage"] == 1.0
    assert inspection.window_frame.iloc[0]["strong_beat_chord_tone_coverage"] == 1.0
    assert inspection.note_frame["chord_tone"].tolist() == [True, True, True, True]
    assert inspection.note_frame["strong_beat"].tolist() == [True, True, True, True]
    assert inspection.summary_rows == [
        {"Metric": "Decoded chord windows", "Value": "1"},
        {"Metric": "Decoded note events", "Value": "4"},
        {"Metric": "Note-event chord tones", "Value": "100.0% (4/4)"},
        {"Metric": "Strong-beat chord tones", "Value": "100.0% (4/4)"},
        {"Metric": "Strong-beat non-chord notes", "Value": "0"},
        {"Metric": "Triadic coincident-pair consonance", "Value": "100.0% (3/3)"},
        {"Metric": "Perfect coincident-pair consonance", "Value": "66.7% (2/3)"},
    ]


def _segment(duration_vocabulary: DurationVocabulary) -> Segment:
    whole_id = duration_vocabulary.require_duration_id(Fraction(1))
    return Segment(
        tokens=[
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=whole_id),
            NoteToken(degree=3, accidental=0, octave_offset=0, duration_id=whole_id),
            JoinWithPreviousToken(),
            NoteToken(degree=5, accidental=0, octave_offset=0, duration_id=whole_id),
            JoinWithPreviousToken(),
            HandToken(hand=Hand.LEFT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=whole_id),
            BarToken(),
            EndToken(),
        ],
        metadata=SegmentMetadata(
            scale_root=0,
            scale_type=ScaleType.MAJOR,
            tokenization_context=tokenization_context_from_scale(scale_root=0, scale_type=ScaleType.MAJOR),
            time_numerator=4,
            time_denominator=4,
            bar_count=1,
            window_start_bar=2,
            source_file=Path("sample.mxl"),
        ),
    )
