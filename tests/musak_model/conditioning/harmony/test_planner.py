from fractions import Fraction

from numpy.random import default_rng

from musak_model.conditioning.harmony.planner import (
    HarmonicPlannerCadenceScores,
    HarmonicPlannerConfig,
    HarmonicPlannerScoreWeights,
    harmonic_plan_candidate_chords,
    harmonic_plan_slots,
    harmonic_slot_roles,
    plan_harmony,
)
from musak_model.conditioning.harmony.schema import HarmonicSlotRole
from musak_model.generation.constraints import GenerationConstraints
from musak_model.harmony.schema import ChordExtension, ChordQuality
from musak_model.harmony.vocabulary import ChordVocabularyConfig, ExtensionDefinition, QualityDefinition
from musak_model.synthetic.processes.chord_track import functional_transition_model, uniform_transition_model
from musak_model.tokens.schema import ScaleType
from musak_shared.elements import HarmonicFunction


def test_harmonic_slot_roles_cover_short_horizons() -> None:
    assert harmonic_slot_roles(1) == (HarmonicSlotRole.CADENCE,)
    assert harmonic_slot_roles(2) == (
        HarmonicSlotRole.CADENCE_PREPARATION,
        HarmonicSlotRole.CADENCE,
    )
    assert harmonic_slot_roles(3) == (
        HarmonicSlotRole.OPENING,
        HarmonicSlotRole.CADENCE_PREPARATION,
        HarmonicSlotRole.CADENCE,
    )
    assert harmonic_slot_roles(4) == (
        HarmonicSlotRole.OPENING,
        HarmonicSlotRole.CADENCE_PREPARATION,
        HarmonicSlotRole.CADENCE_PREPARATION,
        HarmonicSlotRole.CADENCE,
    )


def test_harmonic_plan_slots_cover_requested_span_with_pickup_duration() -> None:
    slots = harmonic_plan_slots(
        constraints=GenerationConstraints(
            time_numerator=4,
            time_denominator=4,
            bar_count=2,
            bar_durations=(Fraction(1, 2), Fraction(1)),
        ),
        harmonic_resolution=1,
    )

    assert slots[0].start == Fraction(0)
    assert slots[0].end == Fraction(1, 2)
    assert slots[1].start == Fraction(1, 2)
    assert slots[1].end == Fraction(3, 2)


def test_two_bar_plan_prefers_dominant_to_tonic_cadence() -> None:
    chord_vocabulary = _chord_vocabulary()
    candidates = harmonic_plan_candidate_chords(
        chord_vocabulary,
        scale_type=ScaleType.MAJOR,
        candidate_limit=16,
    )
    transition_model = functional_transition_model(
        candidates,
        scale_type=ScaleType.MAJOR,
        strength=0.8,
        self_transition_bias=0.0,
    )

    plan = plan_harmony(
        constraints=GenerationConstraints(
            time_numerator=4,
            time_denominator=4,
            bar_count=2,
        ),
        scale_type=ScaleType.MAJOR,
        chord_vocabulary=chord_vocabulary,
        transition_model=transition_model,
        config=_planner_config(sample_from_top_k_plans=1),
        rng=default_rng(0),
    )

    assert tuple(window.slot_role for window in plan.windows) == (
        HarmonicSlotRole.CADENCE_PREPARATION,
        HarmonicSlotRole.CADENCE,
    )
    assert tuple(window.harmonic_function for window in plan.windows) == (
        HarmonicFunction.DOMINANT,
        HarmonicFunction.TONIC,
    )


def test_stasis_weight_changes_the_selected_plan() -> None:
    chord_vocabulary = _chord_vocabulary()
    candidates = harmonic_plan_candidate_chords(
        chord_vocabulary,
        scale_type=ScaleType.MAJOR,
        candidate_limit=16,
    )
    transition_model = uniform_transition_model(candidates)
    constraints = GenerationConstraints(time_numerator=4, time_denominator=4, bar_count=2)

    without_stasis = plan_harmony(
        constraints=constraints,
        scale_type=ScaleType.MAJOR,
        chord_vocabulary=chord_vocabulary,
        transition_model=transition_model,
        config=_planner_config(
            weights=_focused_stasis_weights(stasis=0.0),
            sample_from_top_k_plans=1,
        ),
        rng=default_rng(0),
    )
    with_stasis = plan_harmony(
        constraints=constraints,
        scale_type=ScaleType.MAJOR,
        chord_vocabulary=chord_vocabulary,
        transition_model=transition_model,
        config=_planner_config(
            weights=_focused_stasis_weights(stasis=4.0),
            sample_from_top_k_plans=1,
        ),
        rng=default_rng(0),
    )

    assert without_stasis.windows[0].chord == without_stasis.windows[1].chord
    assert with_stasis.windows[0].chord != with_stasis.windows[1].chord


def test_plan_sampling_is_deterministic_for_seed() -> None:
    chord_vocabulary = _chord_vocabulary()
    candidates = harmonic_plan_candidate_chords(
        chord_vocabulary,
        scale_type=ScaleType.MAJOR,
        candidate_limit=16,
    )
    transition_model = uniform_transition_model(candidates)
    constraints = GenerationConstraints(time_numerator=4, time_denominator=4, bar_count=4)
    config = _planner_config(sample_from_top_k_plans=4)

    first = plan_harmony(
        constraints=constraints,
        scale_type=ScaleType.MAJOR,
        chord_vocabulary=chord_vocabulary,
        transition_model=transition_model,
        config=config,
        rng=default_rng(7),
    )
    second = plan_harmony(
        constraints=constraints,
        scale_type=ScaleType.MAJOR,
        chord_vocabulary=chord_vocabulary,
        transition_model=transition_model,
        config=config,
        rng=default_rng(7),
    )

    assert first.windows == second.windows
    assert len(first.alternatives) == config.alternatives_to_log


def _planner_config(
    *,
    weights: HarmonicPlannerScoreWeights | None = None,
    sample_from_top_k_plans: int = 1,
) -> HarmonicPlannerConfig:
    return HarmonicPlannerConfig(
        harmonic_resolution=1,
        beam_size=64,
        candidate_limit_per_slot=16,
        alternatives_to_log=4,
        sample_from_top_k_plans=sample_from_top_k_plans,
        plan_temperature=0.8,
        seventh_prior_penalty=0.7,
        weights=weights or _default_weights(),
        cadence_scores=HarmonicPlannerCadenceScores(
            final_tonic=2.0,
            final_non_tonic=-1.0,
            dominant_to_tonic_final=1.25,
            predominant_to_dominant_preparation=0.75,
            continuation_to_tonic=0.1,
        ),
    )


def _default_weights() -> HarmonicPlannerScoreWeights:
    return HarmonicPlannerScoreWeights(
        prior=0.5,
        role=0.8,
        cadence=1.5,
        tension=0.25,
        extension=0.3,
        empirical_transition=0.6,
        functional_transition=0.8,
        root_motion=0.2,
        stasis=0.25,
        cadence_approach=1.0,
        terminal=2.0,
        repetition=0.25,
        diversity=0.2,
        shape=0.3,
    )


def _focused_stasis_weights(*, stasis: float) -> HarmonicPlannerScoreWeights:
    return HarmonicPlannerScoreWeights(
        prior=0.0,
        role=0.0,
        cadence=0.0,
        tension=0.0,
        extension=0.0,
        empirical_transition=0.0,
        functional_transition=0.0,
        root_motion=0.0,
        stasis=stasis,
        cadence_approach=0.0,
        terminal=10.0,
        repetition=0.0,
        diversity=0.0,
        shape=0.0,
    )


def _chord_vocabulary() -> ChordVocabularyConfig:
    return ChordVocabularyConfig(
        qualities={
            ChordQuality.MAJOR: QualityDefinition(intervals=(0, 4, 7)),
            ChordQuality.MINOR: QualityDefinition(intervals=(0, 3, 7)),
            ChordQuality.DIMINISHED: QualityDefinition(intervals=(0, 3, 6)),
            ChordQuality.AUGMENTED: QualityDefinition(intervals=(0, 4, 8)),
        },
        extensions={
            ChordExtension.TRIAD: ExtensionDefinition(additional_intervals=()),
            ChordExtension.SEVENTH: ExtensionDefinition(additional_intervals=(10,), enabled=False),
            ChordExtension.MAJOR_SEVENTH: ExtensionDefinition(additional_intervals=(11,), enabled=False),
        },
    )
