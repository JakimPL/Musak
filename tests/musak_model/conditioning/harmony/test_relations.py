from fractions import Fraction

import pytest

from musak_model.conditioning.harmony.relations import (
    HARMONIC_RELATION_IGNORE_ID,
    HarmonicRelationId,
    harmonic_relation_id_for_token,
    harmonic_relation_target_tensors_from_tokens,
)
from musak_model.conditioning.harmony.schema import HarmonicPlanWindow
from musak_model.generation.constraints import GenerationConstraints
from musak_model.generation.coordinates import DecoderInputCoordinates
from musak_model.harmony.schema import Chord, ChordExtension, ChordQuality
from musak_model.harmony.vocabulary import ChordVocabularyConfig
from musak_model.tokens.schema import BarToken, NoteToken, ScaleType


@pytest.mark.parametrize(
    ("token", "chord", "expected_relation"),
    [
        (
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=0),
            Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR),
            HarmonicRelationId.CHORD_ROOT,
        ),
        (
            NoteToken(degree=3, accidental=0, octave_offset=0, duration_id=0),
            Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR),
            HarmonicRelationId.CHORD_THIRD,
        ),
        (
            NoteToken(degree=5, accidental=0, octave_offset=0, duration_id=0),
            Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR),
            HarmonicRelationId.CHORD_FIFTH,
        ),
        (
            NoteToken(degree=7, accidental=-1, octave_offset=0, duration_id=0),
            Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR, extension=ChordExtension.SEVENTH),
            HarmonicRelationId.CHORD_SEVENTH,
        ),
        (
            NoteToken(degree=2, accidental=0, octave_offset=0, duration_id=0),
            Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR),
            HarmonicRelationId.DIATONIC_NON_CHORD,
        ),
        (
            NoteToken(degree=2, accidental=-1, octave_offset=0, duration_id=0),
            Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR),
            HarmonicRelationId.CHROMATIC_NEIGHBOR,
        ),
        (
            NoteToken(degree=7, accidental=-1, octave_offset=0, duration_id=0),
            Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR),
            HarmonicRelationId.OTHER_CHROMATIC,
        ),
    ],
)
def test_harmonic_relation_id_for_token_classifies_note_against_active_chord(
    token: NoteToken,
    chord: Chord,
    expected_relation: HarmonicRelationId,
) -> None:
    relation_id = harmonic_relation_id_for_token(
        token,
        window=HarmonicPlanWindow(start=Fraction(0), end=Fraction(1), chord=chord),
        scale_type=ScaleType.MAJOR,
        chord_vocabulary=ChordVocabularyConfig.load(),
    )

    assert relation_id == expected_relation


def test_harmonic_relation_id_ignores_non_notes_and_missing_windows() -> None:
    chord_vocabulary = ChordVocabularyConfig.load()

    assert (
        harmonic_relation_id_for_token(
            BarToken(),
            window=HarmonicPlanWindow(
                start=Fraction(0),
                end=Fraction(1),
                chord=Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR),
            ),
            scale_type=ScaleType.MAJOR,
            chord_vocabulary=chord_vocabulary,
        )
        == HARMONIC_RELATION_IGNORE_ID
    )
    assert (
        harmonic_relation_id_for_token(
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=0),
            window=None,
            scale_type=ScaleType.MAJOR,
            chord_vocabulary=chord_vocabulary,
        )
        == HARMONIC_RELATION_IGNORE_ID
    )


def test_harmonic_relation_targets_align_tokens_to_decoder_coordinates() -> None:
    targets = harmonic_relation_target_tensors_from_tokens(
        [
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=0),
            NoteToken(degree=2, accidental=0, octave_offset=0, duration_id=0),
            BarToken(),
        ],
        windows=(
            HarmonicPlanWindow(
                start=Fraction(0),
                end=Fraction(1),
                chord=Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR),
            ),
        ),
        constraints=GenerationConstraints(time_numerator=4, time_denominator=4, bar_count=1),
        coordinates=DecoderInputCoordinates(
            bar_indices=(0, 0, 0),
            bar_relative_ticks=(0, 1, 2),
            bar_duration_ticks=(4, 4, 4),
            active_hand_ids=(0, 0, 0),
        ),
        scale_type=ScaleType.MAJOR,
        chord_vocabulary=ChordVocabularyConfig.load(),
        duration_tick_denominator=4,
    )

    assert targets.relation_ids.tolist() == [
        HarmonicRelationId.CHORD_ROOT,
        HarmonicRelationId.DIATONIC_NON_CHORD,
        HARMONIC_RELATION_IGNORE_ID,
    ]
