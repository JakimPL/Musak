from fractions import Fraction

import torch

from musak_model.conditioning.harmony.schema import HarmonicPlanIds, HarmonicPlanWindow, HarmonicSlotRole
from musak_model.conditioning.harmony.vocabulary import (
    CADENCE_STRENGTH_VOCABULARY_SIZE,
    CHORD_CHANGE_VOCABULARY_SIZE,
    CHORD_EXTENSION_VOCABULARY_SIZE,
    CHORD_QUALITY_VOCABULARY_SIZE,
    DISTANCE_TO_END_VOCABULARY_SIZE,
    HARMONIC_FUNCTION_VOCABULARY_SIZE,
    HARMONIC_PLAN_UNKNOWN_ID,
    PLAN_CONFIDENCE_VOCABULARY_SIZE,
    REMAINING_BAR_VOCABULARY_SIZE,
    REMAINING_HARMONIC_SLOT_VOCABULARY_SIZE,
    ROOT_ACCIDENTAL_VOCABULARY_SIZE,
    ROOT_DEGREE_VOCABULARY_SIZE,
    SLOT_ROLE_VOCABULARY_SIZE,
    TENSION_LEVEL_VOCABULARY_SIZE,
    cadence_strength_to_id,
    chord_change_to_id,
    chord_extension_to_id,
    chord_quality_to_id,
    distance_to_end_to_id,
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
    id_to_slot_role,
    plan_confidence_to_id,
    remaining_bar_count_to_id,
    remaining_harmonic_slot_count_to_id,
    root_accidental_to_id,
    root_degree_to_id,
    slot_role_to_id,
    tension_level_to_id,
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
    assert SLOT_ROLE_VOCABULARY_SIZE == len(HarmonicSlotRole) + 1
    assert DISTANCE_TO_END_VOCABULARY_SIZE == REMAINING_HARMONIC_SLOT_VOCABULARY_SIZE
    assert CADENCE_STRENGTH_VOCABULARY_SIZE == TENSION_LEVEL_VOCABULARY_SIZE == PLAN_CONFIDENCE_VOCABULARY_SIZE
    assert REMAINING_BAR_VOCABULARY_SIZE < REMAINING_HARMONIC_SLOT_VOCABULARY_SIZE


def test_unknown_harmonic_plan_ids_use_unknown_bucket_for_every_attribute() -> None:
    assert unknown_harmonic_plan_ids() == HarmonicPlanIds(
        harmonic_function_id=HARMONIC_PLAN_UNKNOWN_ID,
        root_degree_id=HARMONIC_PLAN_UNKNOWN_ID,
        root_accidental_id=HARMONIC_PLAN_UNKNOWN_ID,
        quality_id=HARMONIC_PLAN_UNKNOWN_ID,
        extension_id=HARMONIC_PLAN_UNKNOWN_ID,
        chord_change_id=HARMONIC_PLAN_UNKNOWN_ID,
        slot_role_id=HARMONIC_PLAN_UNKNOWN_ID,
        distance_to_end_id=HARMONIC_PLAN_UNKNOWN_ID,
        cadence_strength_id=HARMONIC_PLAN_UNKNOWN_ID,
        tension_level_id=HARMONIC_PLAN_UNKNOWN_ID,
        plan_confidence_id=HARMONIC_PLAN_UNKNOWN_ID,
        remaining_bar_id=HARMONIC_PLAN_UNKNOWN_ID,
        remaining_harmonic_slot_id=HARMONIC_PLAN_UNKNOWN_ID,
    )


def test_optional_vocabulary_round_trips_known_and_unknown_values() -> None:
    assert id_to_harmonic_function(harmonic_function_to_id(HarmonicFunction.DOMINANT)) == HarmonicFunction.DOMINANT
    assert id_to_root_degree(root_degree_to_id(5)) == 5
    assert id_to_root_accidental(root_accidental_to_id(-1)) == -1
    assert id_to_chord_quality(chord_quality_to_id(ChordQuality.DIMINISHED)) == ChordQuality.DIMINISHED
    assert id_to_chord_extension(chord_extension_to_id(ChordExtension.SEVENTH)) == ChordExtension.SEVENTH
    assert id_to_chord_extension(chord_extension_to_id(ChordExtension.MAJOR_SEVENTH)) == ChordExtension.MAJOR_SEVENTH
    assert id_to_chord_change(chord_change_to_id(True)) is True
    assert id_to_slot_role(slot_role_to_id(HarmonicSlotRole.CADENCE)) == HarmonicSlotRole.CADENCE
    assert distance_to_end_to_id(0) != HARMONIC_PLAN_UNKNOWN_ID
    assert cadence_strength_to_id(0.75) != HARMONIC_PLAN_UNKNOWN_ID
    assert tension_level_to_id(0.85) != HARMONIC_PLAN_UNKNOWN_ID
    assert plan_confidence_to_id(1.0) != HARMONIC_PLAN_UNKNOWN_ID
    assert remaining_bar_count_to_id(3) != HARMONIC_PLAN_UNKNOWN_ID
    assert remaining_harmonic_slot_count_to_id(6) != HARMONIC_PLAN_UNKNOWN_ID

    assert id_to_harmonic_function(HARMONIC_PLAN_UNKNOWN_ID) is None
    assert id_to_root_degree(HARMONIC_PLAN_UNKNOWN_ID) is None
    assert id_to_root_accidental(HARMONIC_PLAN_UNKNOWN_ID) is None
    assert id_to_chord_quality(HARMONIC_PLAN_UNKNOWN_ID) is None
    assert id_to_chord_extension(HARMONIC_PLAN_UNKNOWN_ID) is None
    assert id_to_chord_change(HARMONIC_PLAN_UNKNOWN_ID) is None
    assert id_to_slot_role(HARMONIC_PLAN_UNKNOWN_ID) is None


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
    assert ids.slot_role_id == HARMONIC_PLAN_UNKNOWN_ID
    assert ids.remaining_harmonic_slot_id == HARMONIC_PLAN_UNKNOWN_ID


def test_harmonic_plan_ids_from_windows_marks_first_window_and_chord_changes() -> None:
    tonic = Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR)
    dominant = Chord(root_degree=5, root_accidental=0, quality=ChordQuality.MAJOR)
    windows = (
        HarmonicPlanWindow(
            start=Fraction(0),
            end=Fraction(1, 4),
            chord=tonic,
            slot_role=HarmonicSlotRole.OPENING,
            distance_to_end=2,
            cadence_strength=0.15,
            tension_level=0.0,
            plan_confidence=1.0,
        ),
        HarmonicPlanWindow(
            start=Fraction(1, 4),
            end=Fraction(1, 2),
            chord=tonic,
            slot_role=HarmonicSlotRole.CADENCE_PREPARATION,
            distance_to_end=1,
            cadence_strength=0.75,
            tension_level=0.85,
            plan_confidence=1.0,
        ),
        HarmonicPlanWindow(
            start=Fraction(1, 2),
            end=Fraction(3, 4),
            chord=dominant,
            slot_role=HarmonicSlotRole.CADENCE,
            distance_to_end=0,
            cadence_strength=1.0,
            tension_level=0.0,
            plan_confidence=1.0,
        ),
    )

    ids = harmonic_plan_ids_from_windows(windows)

    assert tuple(id_to_chord_change(item.chord_change_id) for item in ids) == (True, False, True)
    assert tuple(id_to_harmonic_function(item.harmonic_function_id) for item in ids) == (
        HarmonicFunction.TONIC,
        HarmonicFunction.TONIC,
        HarmonicFunction.DOMINANT,
    )
    assert tuple(id_to_slot_role(item.slot_role_id) for item in ids) == (
        HarmonicSlotRole.OPENING,
        HarmonicSlotRole.CADENCE_PREPARATION,
        HarmonicSlotRole.CADENCE,
    )
    assert tuple(item.distance_to_end_id for item in ids) == tuple(distance_to_end_to_id(value) for value in (2, 1, 0))
    assert tuple(item.remaining_harmonic_slot_id for item in ids) == tuple(
        remaining_harmonic_slot_count_to_id(value) for value in (2, 1, 0)
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
            slot_role_id=slot_role_to_id(HarmonicSlotRole.CADENCE),
            distance_to_end_id=distance_to_end_to_id(0),
            cadence_strength_id=cadence_strength_to_id(1.0),
            tension_level_id=tension_level_to_id(0.0),
            plan_confidence_id=plan_confidence_to_id(1.0),
            remaining_bar_id=remaining_bar_count_to_id(0),
            remaining_harmonic_slot_id=remaining_harmonic_slot_count_to_id(0),
        ),
    )

    tensors = harmonic_plan_tensors_from_ids(ids)

    assert tensors.shape == torch.Size([2])
    assert tensors.harmonic_function_ids.dtype == torch.long
    assert torch.equal(
        tensors.root_degree_ids,
        torch.tensor([HARMONIC_PLAN_UNKNOWN_ID, root_degree_to_id(4)]),
    )
    assert torch.equal(
        tensors.slot_role_ids,
        torch.tensor([HARMONIC_PLAN_UNKNOWN_ID, slot_role_to_id(HarmonicSlotRole.CADENCE)]),
    )
    assert torch.equal(
        tensors.remaining_bar_ids,
        torch.tensor([HARMONIC_PLAN_UNKNOWN_ID, remaining_bar_count_to_id(0)]),
    )
    assert torch.equal(tensors.to(torch.device("cpu")).quality_ids, tensors.quality_ids)
