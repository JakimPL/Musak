from fractions import Fraction
from pathlib import Path

import pytest

from musak_model.data.labeler import (
    _has_accidentals,
    _has_dotted_notes,
    _has_note_duration_in,
    _max_hand_span,
    _notes_per_beat,
    _rhythmic_diversity,
    _voice_independence,
    extract_difficulty_features,
)
from musak_model.data.schema import (
    ParsedBar,
    ParsedChord,
    ParsedNote,
    ParsedScore,
    Segment,
    SegmentMetadata,
)
from musak_model.data.tokenization_context import tokenization_context_from_scale
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import BarToken, EndToken, Hand, HandToken, NoteToken, ScaleType, Token
from musak_shared.elements import DOTTED_LIKE_DURATIONS


def _segment_with_tokens(
    duration_vocabulary: DurationVocabulary,
    right_tokens: list[Token] | None = None,
    left_tokens: list[Token] | None = None,
    *,
    window_start_bar: int = 0,
) -> Segment:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))

    if right_tokens is None:
        right_tokens = [
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
            BarToken(),
            EndToken(),
        ]

    if left_tokens is None:
        left_tokens = [
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
            BarToken(),
            EndToken(),
        ]

    return Segment(
        tokens=[HandToken(hand=Hand.RIGHT), *right_tokens, HandToken(hand=Hand.LEFT), *left_tokens],
        metadata=SegmentMetadata(
            scale_root=0,
            scale_type=ScaleType.MAJOR,
            tokenization_context=tokenization_context_from_scale(scale_root=0, scale_type=ScaleType.MAJOR),
            time_numerator=4,
            time_denominator=4,
            bar_count=1,
            window_start_bar=window_start_bar,
            source_file=Path("/test/score.musicxml"),
            difficulty_level=1,
        ),
    )


class TestMaxHandSpan:
    def test_single_note_no_span(self) -> None:
        bars = [
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[ParsedNote(midi_pitch=60, duration=Fraction(1, 4), beat_offset=Fraction(1, 4))],
            )
        ]
        span = _max_hand_span(bars)
        assert span == 0

    def test_two_notes_octave_apart(self) -> None:
        bars = [
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[
                    ParsedNote(midi_pitch=60, duration=Fraction(1, 4), beat_offset=Fraction(1, 8)),
                    ParsedNote(midi_pitch=72, duration=Fraction(1, 4), beat_offset=Fraction(1, 4)),
                ],
            )
        ]
        span = _max_hand_span(bars)
        assert span == 12

    def test_chord_span(self) -> None:
        bars = [
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[
                    ParsedChord(
                        midi_pitches=[60, 64, 67],
                        duration=Fraction(1, 4),
                        beat_offset=Fraction(1, 4),
                    )
                ],
            )
        ]
        span = _max_hand_span(bars)
        assert span == 7

    def test_multiple_bars_max_span(self) -> None:
        bars = [
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[ParsedNote(midi_pitch=60, duration=Fraction(1, 4), beat_offset=Fraction(1, 4))],
            ),
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[
                    ParsedNote(midi_pitch=60, duration=Fraction(1, 4), beat_offset=Fraction(1, 8)),
                    ParsedNote(midi_pitch=80, duration=Fraction(1, 4), beat_offset=Fraction(1, 4)),
                ],
            ),
        ]
        span = _max_hand_span(bars)
        assert span == 20


class TestNotesPerBeat:
    def test_single_note_in_4_4_time(self) -> None:
        bars = [
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[ParsedNote(midi_pitch=60, duration=Fraction(1, 4), beat_offset=Fraction(1, 4))],
            )
        ]
        score = ParsedScore(
            scale_root=0,
            key_fifths=0,
            scale_type=ScaleType.MAJOR,
            time_numerator=4,
            time_denominator=4,
            right_hand_bars=bars,
            left_hand_bars=[],
        )
        npb = _notes_per_beat(bars)
        assert npb == 0.25

    def test_four_notes_in_4_4_time(self) -> None:
        bars = [
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[
                    ParsedNote(midi_pitch=60, duration=Fraction(1, 4), beat_offset=Fraction(1, 8)),
                    ParsedNote(midi_pitch=62, duration=Fraction(1, 4), beat_offset=Fraction(1, 4)),
                    ParsedNote(midi_pitch=64, duration=Fraction(1, 4), beat_offset=Fraction(2, 4)),
                    ParsedNote(midi_pitch=65, duration=Fraction(1, 4), beat_offset=Fraction(3, 4)),
                ],
            )
        ]
        score = ParsedScore(
            scale_root=0,
            key_fifths=0,
            scale_type=ScaleType.MAJOR,
            time_numerator=4,
            time_denominator=4,
            right_hand_bars=bars,
            left_hand_bars=[],
        )
        npb = _notes_per_beat(bars)
        assert npb == 1.0


class TestRhythmicDiversity:
    def test_single_duration_zero_diversity(self, duration_vocabulary: DurationVocabulary) -> None:
        quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))

        segment = _segment_with_tokens(
            duration_vocabulary,
            right_tokens=[
                NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
                NoteToken(degree=2, accidental=0, octave_offset=0, duration_id=quarter_id),
                BarToken(),
                EndToken(),
            ],
        )

        diversity = _rhythmic_diversity(segment, duration_vocabulary=duration_vocabulary)
        assert diversity == pytest.approx(1.0 / duration_vocabulary.vocabulary_size())

    def test_multiple_durations_higher_diversity(self, duration_vocabulary: DurationVocabulary) -> None:
        quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
        eighth_id = duration_vocabulary.fraction_to_id(Fraction(1, 8))

        segment = _segment_with_tokens(
            duration_vocabulary,
            right_tokens=[
                NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
                NoteToken(degree=2, accidental=0, octave_offset=0, duration_id=eighth_id),
                BarToken(),
                EndToken(),
            ],
        )

        diversity = _rhythmic_diversity(segment, duration_vocabulary=duration_vocabulary)
        assert diversity == pytest.approx(2.0 / duration_vocabulary.vocabulary_size())


class TestVoiceIndependence:
    def test_identical_rhythm_zero_independence(self, duration_vocabulary: DurationVocabulary) -> None:
        quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))

        segment = _segment_with_tokens(
            duration_vocabulary,
            right_tokens=[
                NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
                BarToken(),
                EndToken(),
            ],
            left_tokens=[
                NoteToken(degree=5, accidental=0, octave_offset=0, duration_id=quarter_id),
                BarToken(),
                EndToken(),
            ],
        )

        independence = _voice_independence(segment, duration_vocabulary=duration_vocabulary)
        assert independence == 0.0

    def test_different_rhythms_nonzero_independence(self, duration_vocabulary: DurationVocabulary) -> None:
        quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
        eighth_id = duration_vocabulary.fraction_to_id(Fraction(1, 8))

        segment = _segment_with_tokens(
            duration_vocabulary,
            right_tokens=[
                NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
                BarToken(),
                EndToken(),
            ],
            left_tokens=[
                NoteToken(degree=5, accidental=0, octave_offset=0, duration_id=eighth_id),
                BarToken(),
                EndToken(),
            ],
        )

        independence = _voice_independence(segment, duration_vocabulary=duration_vocabulary)
        assert independence > 0.0


class TestHasAccidentals:
    def test_no_accidentals_in_c_major(self) -> None:
        bars = [
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[
                    ParsedNote(midi_pitch=60, duration=Fraction(1, 4), beat_offset=Fraction(1, 8)),  # C
                    ParsedNote(midi_pitch=64, duration=Fraction(1, 4), beat_offset=Fraction(1, 4)),  # E
                    ParsedNote(midi_pitch=67, duration=Fraction(1, 4), beat_offset=Fraction(2, 4)),  # G
                ],
            )
        ]
        segment = Segment(
            tokens=[],
            metadata=SegmentMetadata(
                scale_root=0,
                scale_type=ScaleType.MAJOR,
                tokenization_context=tokenization_context_from_scale(scale_root=0, scale_type=ScaleType.MAJOR),
                time_numerator=4,
                time_denominator=4,
                bar_count=1,
                window_start_bar=0,
                source_file=Path("/test/score.musicxml"),
            ),
        )

        has_acc = _has_accidentals(bars, segment=segment, hand=Hand.RIGHT)
        assert has_acc is False

    def test_accidental_detected_in_chromatic_note(self) -> None:
        bars = [
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[
                    ParsedNote(midi_pitch=61, duration=Fraction(1, 4), beat_offset=Fraction(1, 4)),  # C#
                ],
            )
        ]
        segment = Segment(
            tokens=[],
            metadata=SegmentMetadata(
                scale_root=0,
                scale_type=ScaleType.MAJOR,
                tokenization_context=tokenization_context_from_scale(scale_root=0, scale_type=ScaleType.MAJOR),
                time_numerator=4,
                time_denominator=4,
                bar_count=1,
                window_start_bar=0,
                source_file=Path("/test/score.musicxml"),
            ),
        )

        has_acc = _has_accidentals(bars, segment=segment, hand=Hand.RIGHT)
        assert has_acc is True


class TestHasDottedNotes:
    def test_no_dotted_notes(self, duration_vocabulary: DurationVocabulary) -> None:
        half_id = duration_vocabulary.fraction_to_id(Fraction(1, 2))

        segment = _segment_with_tokens(
            duration_vocabulary,
            right_tokens=[
                NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=half_id),
                BarToken(),
                EndToken(),
            ],
        )

        has_dotted = _has_dotted_notes(segment, duration_vocabulary=duration_vocabulary)
        assert has_dotted is False

    def test_dotted_note_detected(self, duration_vocabulary: DurationVocabulary) -> None:
        dotted_quarter_id = duration_vocabulary.fraction_to_id(Fraction(3, 8))

        segment = _segment_with_tokens(
            duration_vocabulary,
            right_tokens=[
                NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=dotted_quarter_id),
                BarToken(),
                EndToken(),
            ],
        )

        has_dotted = _has_dotted_notes(segment, duration_vocabulary=duration_vocabulary)
        assert has_dotted is True

    def test_duration_membership_helper_uses_explicit_fraction_set(
        self,
        duration_vocabulary: DurationVocabulary,
    ) -> None:
        eighth_id = duration_vocabulary.fraction_to_id(Fraction(1, 8))

        segment = _segment_with_tokens(
            duration_vocabulary,
            right_tokens=[
                NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=eighth_id),
                BarToken(),
                EndToken(),
            ],
        )

        assert (
            _has_note_duration_in(
                segment,
                duration_vocabulary=duration_vocabulary,
                durations=frozenset({Fraction(1, 8)}),
            )
            is True
        )
        assert (
            _has_note_duration_in(
                segment,
                duration_vocabulary=duration_vocabulary,
                durations=DOTTED_LIKE_DURATIONS,
            )
            is False
        )


class TestExtractDifficultyFeaturesIntegration:
    def test_all_features_computed(self, duration_vocabulary: DurationVocabulary) -> None:
        quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))

        segment = _segment_with_tokens(
            duration_vocabulary,
            right_tokens=[
                NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
                BarToken(),
                EndToken(),
            ],
        )

        score = ParsedScore(
            scale_root=0,
            key_fifths=0,
            scale_type=ScaleType.MAJOR,
            time_numerator=4,
            time_denominator=4,
            right_hand_bars=[
                ParsedBar(
                    time_numerator=4,
                    time_denominator=4,
                    key_fifths=0,
                    events=[ParsedNote(midi_pitch=60, duration=Fraction(1, 4), beat_offset=Fraction(1, 4))],
                )
            ],
            left_hand_bars=[ParsedBar(time_numerator=4, time_denominator=4, key_fifths=0, events=[])],
        )

        features = extract_difficulty_features(
            segment, score=score, scale_type=ScaleType.MAJOR, duration_vocabulary=duration_vocabulary
        )

        assert hasattr(features, "max_right_hand_span_semitones")
        assert hasattr(features, "max_left_hand_span_semitones")
        assert hasattr(features, "notes_per_beat")
        assert hasattr(features, "rhythmic_diversity")
        assert hasattr(features, "voice_independence")
        assert hasattr(features, "has_accidentals")
        assert hasattr(features, "has_dotted_notes")

        assert 0 <= features.rhythmic_diversity <= 1.0
        assert 0 <= features.voice_independence <= 1.0
        assert isinstance(features.has_accidentals, bool)
        assert isinstance(features.has_dotted_notes, bool)

    def test_left_hand_accidental_check_uses_left_hand_register(self, duration_vocabulary: DurationVocabulary) -> None:
        quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
        segment = _segment_with_tokens(
            duration_vocabulary,
            left_tokens=[
                NoteToken(degree=1, accidental=0, octave_offset=-1, duration_id=quarter_id),
                BarToken(),
                EndToken(),
            ],
        )
        score = ParsedScore(
            scale_root=0,
            key_fifths=0,
            scale_type=ScaleType.MAJOR,
            time_numerator=4,
            time_denominator=4,
            right_hand_bars=[ParsedBar(time_numerator=4, time_denominator=4, key_fifths=0, events=[])],
            left_hand_bars=[
                ParsedBar(
                    time_numerator=4,
                    time_denominator=4,
                    key_fifths=0,
                    events=[ParsedNote(midi_pitch=36, duration=Fraction(1, 4), beat_offset=Fraction(0))],
                )
            ],
        )

        features = extract_difficulty_features(
            segment,
            score=score,
            scale_type=ScaleType.MAJOR,
            duration_vocabulary=duration_vocabulary,
        )

        assert features.has_accidentals is False
