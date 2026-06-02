from fractions import Fraction

import torch

from musak_model.conditioning.harmony.schema import HarmonicPlanIds, HarmonicPlanWindow
from musak_model.conditioning.harmony.vocabulary import (
    CHORD_CHANGE_VOCABULARY_SIZE,
    CHORD_EXTENSION_VOCABULARY_SIZE,
    CHORD_QUALITY_VOCABULARY_SIZE,
    HARMONIC_FUNCTION_VOCABULARY_SIZE,
    HARMONIC_PLAN_UNKNOWN_ID,
    ROOT_ACCIDENTAL_VOCABULARY_SIZE,
    ROOT_DEGREE_VOCABULARY_SIZE,
    chord_change_to_id,
    chord_extension_to_id,
    chord_quality_to_id,
    harmonic_function_to_id,
    harmonic_plan_ids_from_chord,
    harmonic_plan_ids_from_windows,
    harmonic_plan_tensors_from_ids,
    id_to_chord_change,
    id_to_chord_extension,
    id_to_chord_quality,
    id_to_harmonic_function,
    id_to_root_accidental,
    id_to_root_degree,
    root_accidental_to_id,
    root_degree_to_id,
    unknown_harmonic_plan_ids,
)
from musak_model.harmony.schema import Chord, ChordExtension, ChordQuality
from musak_model.tokens.schema import MAX_ACCIDENTAL, MAX_DEGREE, MIN_ACCIDENTAL, MIN_DEGREE
from musak_shared.elements import HarmonicFunction


def test_vocabulary_sizes_include_unknown_bucket() -> None:
    assert HARMONIC_FUNCTION_VOCABULARY_SIZE == len(HarmonicFunction) + 1
    assert ROOT_DEGREE_VOCABULARY_SIZE == MAX_DEGREE - MIN_DEGREE + 2
    assert ROOT_ACCIDENTAL_VOCABULARY_SIZE == MAX_ACCIDENTAL - MIN_ACCIDENTAL + 2
    assert CHORD_QUALITY_VOCABULARY_SIZE == len(ChordQuality) + 1
    assert CHORD_EXTENSION_VOCABULARY_SIZE == len(ChordExtension) + 1
    assert CHORD_CHANGE_VOCABULARY_SIZE == len((False, True)) + 1


def test_unknown_harmonic_plan_ids_use_unknown_bucket_for_every_attribute() -> None:
    assert unknown_harmonic_plan_ids() == HarmonicPlanIds(
        harmonic_function_id=HARMONIC_PLAN_UNKNOWN_ID,
        root_degree_id=HARMONIC_PLAN_UNKNOWN_ID,
        root_accidental_id=HARMONIC_PLAN_UNKNOWN_ID,
        quality_id=HARMONIC_PLAN_UNKNOWN_ID,
        extension_id=HARMONIC_PLAN_UNKNOWN_ID,
        chord_change_id=HARMONIC_PLAN_UNKNOWN_ID,
    )


def test_optional_vocabulary_round_trips_known_and_unknown_values() -> None:
    assert id_to_harmonic_function(harmonic_function_to_id(HarmonicFunction.DOMINANT)) == HarmonicFunction.DOMINANT
    assert id_to_root_degree(root_degree_to_id(5)) == 5
    assert id_to_root_accidental(root_accidental_to_id(-1)) == -1
    assert id_to_chord_quality(chord_quality_to_id(ChordQuality.DIMINISHED)) == ChordQuality.DIMINISHED
    assert id_to_chord_extension(chord_extension_to_id(ChordExtension.SEVENTH)) == ChordExtension.SEVENTH
    assert id_to_chord_extension(chord_extension_to_id(ChordExtension.MAJOR_SEVENTH)) == ChordExtension.MAJOR_SEVENTH
    assert id_to_chord_change(chord_change_to_id(True)) is True

    assert id_to_harmonic_function(HARMONIC_PLAN_UNKNOWN_ID) is None
    assert id_to_root_degree(HARMONIC_PLAN_UNKNOWN_ID) is None
    assert id_to_root_accidental(HARMONIC_PLAN_UNKNOWN_ID) is None
    assert id_to_chord_quality(HARMONIC_PLAN_UNKNOWN_ID) is None
    assert id_to_chord_extension(HARMONIC_PLAN_UNKNOWN_ID) is None
    assert id_to_chord_change(HARMONIC_PLAN_UNKNOWN_ID) is None


def test_harmonic_plan_ids_from_chord_derive_all_chord_attributes() -> None:
    ids = harmonic_plan_ids_from_chord(
        Chord(
            root_degree=6,
            root_accidental=-1,
            quality=ChordQuality.MAJOR,
            extension=ChordExtension.SEVENTH,
        ),
        chord_changed=True,
    )

    assert id_to_harmonic_function(ids.harmonic_function_id) == HarmonicFunction.TONIC
    assert id_to_root_degree(ids.root_degree_id) == 6
    assert id_to_root_accidental(ids.root_accidental_id) == -1
    assert id_to_chord_quality(ids.quality_id) == ChordQuality.MAJOR
    assert id_to_chord_extension(ids.extension_id) == ChordExtension.SEVENTH
    assert id_to_chord_change(ids.chord_change_id) is True


def test_harmonic_plan_ids_from_windows_marks_first_window_and_chord_changes() -> None:
    tonic = Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR)
    dominant = Chord(root_degree=5, root_accidental=0, quality=ChordQuality.MAJOR)
    windows = (
        HarmonicPlanWindow(start=Fraction(0), end=Fraction(1, 4), chord=tonic),
        HarmonicPlanWindow(start=Fraction(1, 4), end=Fraction(1, 2), chord=tonic),
        HarmonicPlanWindow(start=Fraction(1, 2), end=Fraction(3, 4), chord=dominant),
    )

    ids = harmonic_plan_ids_from_windows(windows)

    assert tuple(id_to_chord_change(item.chord_change_id) for item in ids) == (True, False, True)
    assert tuple(id_to_harmonic_function(item.harmonic_function_id) for item in ids) == (
        HarmonicFunction.TONIC,
        HarmonicFunction.TONIC,
        HarmonicFunction.DOMINANT,
    )


def test_harmonic_plan_tensors_from_ids_returns_long_tensor_bundle() -> None:
    ids = (
        unknown_harmonic_plan_ids(),
        HarmonicPlanIds(
            harmonic_function_id=harmonic_function_to_id(HarmonicFunction.PREDOMINANT),
            root_degree_id=root_degree_to_id(4),
            root_accidental_id=root_accidental_to_id(0),
            quality_id=chord_quality_to_id(ChordQuality.MAJOR),
            extension_id=chord_extension_to_id(ChordExtension.TRIAD),
            chord_change_id=chord_change_to_id(False),
        ),
    )

    tensors = harmonic_plan_tensors_from_ids(ids)

    assert tensors.shape == torch.Size([2])
    assert tensors.harmonic_function_ids.dtype == torch.long
    assert torch.equal(
        tensors.root_degree_ids,
        torch.tensor([HARMONIC_PLAN_UNKNOWN_ID, root_degree_to_id(4)]),
    )
    assert torch.equal(tensors.to(torch.device("cpu")).quality_ids, tensors.quality_ids)
