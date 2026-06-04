from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import exp, log
from pathlib import Path
from typing import Final, Self

from numpy.random import Generator
from pydantic import BaseModel, ConfigDict, Field, model_validator

from musak_model.conditioning.harmony.schema import HarmonicPlanWindow, HarmonicSlotRole, harmonic_function_for_chord
from musak_model.generation.constraints import GenerationConstraints
from musak_model.harmony.diatonic import diatonic_triads
from musak_model.harmony.schema import Chord, ChordExtension
from musak_model.harmony.vocabulary import ChordVocabularyConfig
from musak_model.paths import HARMONIC_PLANNER_CONFIG_PATH
from musak_model.synthetic.processes.chord_track import ChordTransitionModel
from musak_model.tokens.schema import ScaleType
from musak_shared.elements import HarmonicFunction
from musak_shared.files import load_yaml_config

_LOG_PROBABILITY_FLOOR: Final[float] = 1e-12
_OPENING_CADENCE_STRENGTH: Final[float] = 0.15
_CONTINUATION_CADENCE_STRENGTH: Final[float] = 0.25
_CADENCE_PREPARATION_STRENGTH: Final[float] = 0.75
_CADENCE_STRENGTH: Final[float] = 1.0
_LOW_TENSION: Final[float] = 0.0
_CONTINUATION_TENSION: Final[float] = 0.35
_PREDOMINANT_TENSION: Final[float] = 0.55
_DOMINANT_TENSION: Final[float] = 0.85
_DEFAULT_PLAN_CONFIDENCE: Final[float] = 1.0
_SAME_CHORD_RUN_GRACE_LENGTH: Final[int] = 1
_ROOT_STEP_MOTION_DISTANCE: Final[int] = 1
_ROOT_THIRD_MOTION_DISTANCE: Final[int] = 2
_ROOT_FOURTH_OR_FIFTH_MOTION_DISTANCE: Final[int] = 3
_ROOT_MOTION_REWARD: Final[float] = 0.25
_ROOT_MOTION_NEUTRAL: Final[float] = 0.0
_ROOT_MOTION_PENALTY: Final[float] = -0.20
_ROLE_STRONG_REWARD: Final[float] = 1.0
_ROLE_MEDIUM_REWARD: Final[float] = 0.60
_ROLE_SMALL_REWARD: Final[float] = 0.25
_ROLE_SMALL_PENALTY: Final[float] = -0.25
_ROLE_MEDIUM_PENALTY: Final[float] = -0.60
_ROLE_STRONG_PENALTY: Final[float] = -1.0


class HarmonicPlannerScoreWeights(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    prior: float
    role: float
    cadence: float
    tension: float
    extension: float
    empirical_transition: float
    functional_transition: float
    root_motion: float
    stasis: float
    cadence_approach: float
    terminal: float
    repetition: float
    diversity: float
    shape: float


class HarmonicPlannerCadenceScores(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    final_tonic: float
    final_non_tonic: float
    dominant_to_tonic_final: float
    predominant_to_dominant_preparation: float
    continuation_to_tonic: float


class HarmonicPlannerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    harmonic_resolution: int = Field(gt=0)
    beam_size: int = Field(gt=0)
    candidate_limit_per_slot: int = Field(gt=0)
    alternatives_to_log: int = Field(gt=0)
    sample_from_top_k_plans: int = Field(gt=0)
    plan_temperature: float = Field(gt=0.0)
    seventh_prior_penalty: float = Field(ge=0.0)
    weights: HarmonicPlannerScoreWeights
    cadence_scores: HarmonicPlannerCadenceScores

    @classmethod
    def load(cls, path: Path = HARMONIC_PLANNER_CONFIG_PATH) -> HarmonicPlannerConfig:
        return cls.model_validate(load_yaml_config(path))

    @model_validator(mode="after")
    def _validate_plan_sampling(self) -> Self:
        if self.sample_from_top_k_plans > self.beam_size:
            raise ValueError("sample_from_top_k_plans cannot exceed beam_size")

        if self.alternatives_to_log > self.beam_size:
            raise ValueError("alternatives_to_log cannot exceed beam_size")

        return self


@dataclass(frozen=True)
class HarmonicPlanSlot:
    index: int
    start: Fraction
    end: Fraction
    role: HarmonicSlotRole
    distance_to_end: int
    cadence_strength: float
    tension_level: float


@dataclass(frozen=True)
class HarmonicPlanAlternative:
    windows: tuple[HarmonicPlanWindow, ...]
    score: float


@dataclass(frozen=True)
class HarmonicPlan:
    windows: tuple[HarmonicPlanWindow, ...]
    score: float
    alternatives: tuple[HarmonicPlanAlternative, ...]


@dataclass(frozen=True)
class _BeamState:
    chords: tuple[Chord, ...]
    score: float


@dataclass(frozen=True)
class _ScoredChordSequence:
    chords: tuple[Chord, ...]
    score: float


def plan_harmony(
    *,
    constraints: GenerationConstraints,
    scale_type: ScaleType,
    chord_vocabulary: ChordVocabularyConfig,
    transition_model: ChordTransitionModel,
    config: HarmonicPlannerConfig,
    rng: Generator,
) -> HarmonicPlan:
    slots = harmonic_plan_slots(
        constraints=constraints,
        harmonic_resolution=config.harmonic_resolution,
    )
    candidates = harmonic_plan_candidate_chords(
        chord_vocabulary,
        scale_type=scale_type,
        candidate_limit=config.candidate_limit_per_slot,
    )
    ranked_sequences = _rank_chord_sequences(
        slots=slots,
        candidates=candidates,
        transition_model=transition_model,
        config=config,
    )
    selected = _sample_ranked_sequence(
        ranked_sequences,
        sample_count=config.sample_from_top_k_plans,
        temperature=config.plan_temperature,
        rng=rng,
    )
    alternatives = tuple(
        HarmonicPlanAlternative(
            windows=_windows_from_chords(
                sequence.chords,
                slots=slots,
                transition_model=transition_model,
                config=config,
            ),
            score=sequence.score,
        )
        for sequence in ranked_sequences[: config.alternatives_to_log]
    )
    windows = _windows_from_chords(
        selected.chords,
        slots=slots,
        transition_model=transition_model,
        config=config,
    )
    validate_harmonic_plan_coverage(windows, constraints=constraints)
    return HarmonicPlan(windows=windows, score=selected.score, alternatives=alternatives)


def harmonic_plan_slots(
    *,
    constraints: GenerationConstraints,
    harmonic_resolution: int,
) -> tuple[HarmonicPlanSlot, ...]:
    if harmonic_resolution <= 0:
        raise ValueError("harmonic_resolution must be positive")

    bounds = harmonic_plan_window_bounds(constraints=constraints, resolution=harmonic_resolution)
    roles = harmonic_slot_roles(len(bounds))
    final_index = len(bounds) - 1
    return tuple(
        HarmonicPlanSlot(
            index=index,
            start=start,
            end=end,
            role=role,
            distance_to_end=final_index - index,
            cadence_strength=_cadence_strength_for_role(role),
            tension_level=_tension_level_for_role(role),
        )
        for index, ((start, end), role) in enumerate(zip(bounds, roles, strict=True))
    )


def harmonic_slot_roles(horizon: int) -> tuple[HarmonicSlotRole, ...]:
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    match horizon:
        case 1:
            return (HarmonicSlotRole.CADENCE,)
        case 2:
            return (HarmonicSlotRole.CADENCE_PREPARATION, HarmonicSlotRole.CADENCE)
        case 3:
            return (
                HarmonicSlotRole.OPENING,
                HarmonicSlotRole.CADENCE_PREPARATION,
                HarmonicSlotRole.CADENCE,
            )

    preparation_count = min(2, horizon - 2)
    continuation_count = horizon - preparation_count - 2
    return (
        HarmonicSlotRole.OPENING,
        *((HarmonicSlotRole.CONTINUATION,) * continuation_count),
        *((HarmonicSlotRole.CADENCE_PREPARATION,) * preparation_count),
        HarmonicSlotRole.CADENCE,
    )


def harmonic_plan_window_bounds(
    *,
    constraints: GenerationConstraints,
    resolution: int,
) -> tuple[tuple[Fraction, Fraction], ...]:
    if resolution <= 0:
        raise ValueError("resolution must be positive")

    window_duration = Fraction(1, resolution)
    windows: list[tuple[Fraction, Fraction]] = []
    for bar_index in range(constraints.bar_count):
        bar_end = constraints.bar_end(bar_index)
        window_start = constraints.bar_start(bar_index)
        while window_start < bar_end:
            window_end = min(window_start + window_duration, bar_end)
            if window_end <= window_start:
                raise ValueError("harmonic plan window bounds must make forward progress")

            windows.append((window_start, window_end))
            window_start = window_end

    return tuple(windows)


def harmonic_plan_candidate_chords(
    chord_vocabulary: ChordVocabularyConfig,
    *,
    scale_type: ScaleType,
    candidate_limit: int,
) -> tuple[Chord, ...]:
    if candidate_limit <= 0:
        raise ValueError("candidate_limit must be positive")

    enabled_qualities = set(chord_vocabulary.enabled_qualities())
    enabled_extensions = chord_vocabulary.enabled_extensions()
    candidates = tuple(
        Chord(
            root_degree=triad.root_degree,
            root_accidental=triad.root_accidental,
            quality=triad.quality,
            extension=extension,
        )
        for triad in diatonic_triads(scale_type)
        if triad.quality in enabled_qualities
        for extension in enabled_extensions
    )
    if not candidates:
        raise ValueError("harmonic planner candidate set must be non-empty")

    return candidates[:candidate_limit]


def validate_harmonic_plan_coverage(
    windows: Sequence[HarmonicPlanWindow],
    *,
    constraints: GenerationConstraints,
) -> None:
    if not windows:
        raise ValueError("harmonic plan windows must be non-empty")

    expected_start = constraints.bar_start(0)
    expected_end = constraints.bar_end(constraints.bar_count - 1)
    if windows[0].start != expected_start:
        raise ValueError(f"harmonic plan must start at {expected_start}, got {windows[0].start}")

    previous_end = windows[0].start
    for window in windows:
        if window.start != previous_end:
            raise ValueError("harmonic plan windows must cover the requested span without gaps")

        previous_end = window.end

    if previous_end != expected_end:
        raise ValueError(f"harmonic plan must end at {expected_end}, got {previous_end}")


def _rank_chord_sequences(
    *,
    slots: tuple[HarmonicPlanSlot, ...],
    candidates: tuple[Chord, ...],
    transition_model: ChordTransitionModel,
    config: HarmonicPlannerConfig,
) -> tuple[_ScoredChordSequence, ...]:
    beams: tuple[_BeamState, ...] = (_BeamState(chords=(), score=0.0),)
    for slot in slots:
        expanded: list[_BeamState] = []
        for beam in beams:
            previous_chord = beam.chords[-1] if beam.chords else None
            for chord in candidates:
                local_score = sum(
                    _local_score_terms(
                        chord,
                        previous_chord=previous_chord,
                        slot=slot,
                        transition_model=transition_model,
                        config=config,
                    ).values()
                )
                expanded.append(_BeamState(chords=(*beam.chords, chord), score=beam.score + local_score))

        beams = tuple(sorted(expanded, key=lambda state: state.score, reverse=True)[: config.beam_size])

    scored_sequences = tuple(
        _ScoredChordSequence(
            chords=beam.chords,
            score=beam.score
            + sum(
                _global_score_terms(
                    beam.chords,
                    slots=slots,
                    config=config,
                ).values()
            ),
        )
        for beam in beams
    )
    return tuple(sorted(scored_sequences, key=lambda sequence: sequence.score, reverse=True))


def _sample_ranked_sequence(
    sequences: tuple[_ScoredChordSequence, ...],
    *,
    sample_count: int,
    temperature: float,
    rng: Generator,
) -> _ScoredChordSequence:
    if not sequences:
        raise ValueError("cannot sample from an empty plan sequence list")

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    top_sequences = sequences[:sample_count]
    best_score = top_sequences[0].score
    weights = tuple(exp((sequence.score - best_score) / temperature) for sequence in top_sequences)
    total = sum(weights)
    probabilities = tuple(weight / total for weight in weights)
    selected_index = int(rng.choice(len(top_sequences), p=probabilities))
    return top_sequences[selected_index]


def _windows_from_chords(
    chords: tuple[Chord, ...],
    *,
    slots: tuple[HarmonicPlanSlot, ...],
    transition_model: ChordTransitionModel,
    config: HarmonicPlannerConfig,
) -> tuple[HarmonicPlanWindow, ...]:
    if len(chords) != len(slots):
        raise ValueError("chord and slot counts must match")

    global_terms = _global_score_terms(chords, slots=slots, config=config)
    windows: list[HarmonicPlanWindow] = []
    for index, (chord, slot) in enumerate(zip(chords, slots, strict=True)):
        previous_chord = chords[index - 1] if index > 0 else None
        score_terms = _local_score_terms(
            chord,
            previous_chord=previous_chord,
            slot=slot,
            transition_model=transition_model,
            config=config,
        )
        if index == len(chords) - 1:
            score_terms = {**score_terms, **global_terms}

        windows.append(
            HarmonicPlanWindow(
                start=slot.start,
                end=slot.end,
                chord=chord,
                slot_role=slot.role,
                distance_to_end=slot.distance_to_end,
                cadence_strength=slot.cadence_strength,
                tension_level=slot.tension_level,
                plan_confidence=_DEFAULT_PLAN_CONFIDENCE,
                score_terms=score_terms,
            )
        )

    return tuple(windows)


def _local_score_terms(
    chord: Chord,
    *,
    previous_chord: Chord | None,
    slot: HarmonicPlanSlot,
    transition_model: ChordTransitionModel,
    config: HarmonicPlannerConfig,
) -> dict[str, float]:
    weights = config.weights
    terms = {
        "prior": weights.prior * _log_probability(transition_model.initial_distribution[chord]),
        "role": weights.role * _role_compatibility_score(chord, slot.role),
        "cadence": weights.cadence * _cadence_compatibility_score(chord, slot=slot, config=config),
        "tension": weights.tension * _tension_curve_score(chord, slot=slot),
        "extension": weights.extension * _extension_prior_score(chord, config=config),
    }
    if previous_chord is None:
        return terms

    terms.update(
        {
            "empirical_transition": 0.0,
            "functional_transition": weights.functional_transition
            * _log_probability(transition_model.transitions[previous_chord][chord]),
            "root_motion": weights.root_motion * _root_motion_score(previous_chord, chord),
            "stasis": weights.stasis * _stasis_score(previous_chord, chord),
            "cadence_approach": weights.cadence_approach
            * _cadence_approach_score(previous_chord, chord, slot=slot, config=config),
        }
    )
    return terms


def _global_score_terms(
    chords: tuple[Chord, ...],
    *,
    slots: tuple[HarmonicPlanSlot, ...],
    config: HarmonicPlannerConfig,
) -> dict[str, float]:
    weights = config.weights
    return {
        "terminal": weights.terminal * _terminal_cadence_score(chords, config=config),
        "repetition": weights.repetition * _repetition_variation_score(chords),
        "diversity": weights.diversity * _harmonic_diversity_score(chords),
        "shape": weights.shape * _tension_shape_score(chords, slots=slots),
    }


def _role_compatibility_score(chord: Chord, role: HarmonicSlotRole) -> float:
    harmonic_function = harmonic_function_for_chord(chord)
    match role:
        case HarmonicSlotRole.OPENING:
            return _function_score(
                harmonic_function,
                tonic=_ROLE_STRONG_REWARD,
                predominant=_ROLE_SMALL_PENALTY,
                dominant=_ROLE_MEDIUM_PENALTY,
            )
        case HarmonicSlotRole.CONTINUATION:
            return _function_score(
                harmonic_function,
                tonic=_ROLE_SMALL_REWARD,
                predominant=_ROLE_SMALL_REWARD,
                dominant=_ROLE_SMALL_REWARD,
            )
        case HarmonicSlotRole.CADENCE_PREPARATION:
            return _function_score(
                harmonic_function,
                tonic=_ROLE_SMALL_PENALTY,
                predominant=_ROLE_STRONG_REWARD,
                dominant=_ROLE_MEDIUM_REWARD,
            )
        case HarmonicSlotRole.CADENCE:
            return _function_score(
                harmonic_function,
                tonic=_ROLE_STRONG_REWARD,
                predominant=_ROLE_MEDIUM_PENALTY,
                dominant=_ROLE_STRONG_PENALTY,
            )


def _cadence_compatibility_score(
    chord: Chord,
    *,
    slot: HarmonicPlanSlot,
    config: HarmonicPlannerConfig,
) -> float:
    harmonic_function = harmonic_function_for_chord(chord)
    match slot.role:
        case HarmonicSlotRole.CADENCE:
            if harmonic_function == HarmonicFunction.TONIC:
                return config.cadence_scores.final_tonic

            return config.cadence_scores.final_non_tonic
        case HarmonicSlotRole.CADENCE_PREPARATION:
            if harmonic_function in {HarmonicFunction.PREDOMINANT, HarmonicFunction.DOMINANT}:
                return config.cadence_scores.predominant_to_dominant_preparation

            return 0.0
        case HarmonicSlotRole.CONTINUATION:
            if harmonic_function == HarmonicFunction.TONIC and slot.distance_to_end > 1:
                return config.cadence_scores.continuation_to_tonic

            return 0.0
        case HarmonicSlotRole.OPENING:
            return 0.0


def _cadence_approach_score(
    previous_chord: Chord,
    chord: Chord,
    *,
    slot: HarmonicPlanSlot,
    config: HarmonicPlannerConfig,
) -> float:
    previous_function = harmonic_function_for_chord(previous_chord)
    harmonic_function = harmonic_function_for_chord(chord)
    match slot.role:
        case HarmonicSlotRole.CADENCE:
            if previous_function == HarmonicFunction.DOMINANT and harmonic_function == HarmonicFunction.TONIC:
                return config.cadence_scores.dominant_to_tonic_final

            return 0.0
        case HarmonicSlotRole.CADENCE_PREPARATION:
            if previous_function == HarmonicFunction.PREDOMINANT and harmonic_function == HarmonicFunction.DOMINANT:
                return config.cadence_scores.predominant_to_dominant_preparation

            return 0.0
        case HarmonicSlotRole.OPENING | HarmonicSlotRole.CONTINUATION:
            return 0.0


def _terminal_cadence_score(chords: tuple[Chord, ...], *, config: HarmonicPlannerConfig) -> float:
    final_function = harmonic_function_for_chord(chords[-1])
    if final_function == HarmonicFunction.TONIC:
        return config.cadence_scores.final_tonic

    return config.cadence_scores.final_non_tonic


def _tension_curve_score(chord: Chord, *, slot: HarmonicPlanSlot) -> float:
    return -abs(_function_tension_level(harmonic_function_for_chord(chord)) - slot.tension_level)


def _tension_shape_score(chords: tuple[Chord, ...], *, slots: tuple[HarmonicPlanSlot, ...]) -> float:
    tension_error = sum(
        abs(_function_tension_level(harmonic_function_for_chord(chord)) - slot.tension_level)
        for chord, slot in zip(chords, slots, strict=True)
    )
    return -tension_error / len(chords)


def _extension_prior_score(chord: Chord, *, config: HarmonicPlannerConfig) -> float:
    if chord.extension == ChordExtension.TRIAD:
        return 0.0

    return -config.seventh_prior_penalty


def _root_motion_score(previous_chord: Chord, chord: Chord) -> float:
    if previous_chord.root_degree == chord.root_degree:
        return _ROOT_MOTION_NEUTRAL

    distance = abs(chord.root_degree - previous_chord.root_degree)
    cyclic_distance = min(distance, 7 - distance)
    if cyclic_distance in {
        _ROOT_STEP_MOTION_DISTANCE,
        _ROOT_THIRD_MOTION_DISTANCE,
        _ROOT_FOURTH_OR_FIFTH_MOTION_DISTANCE,
    }:
        return _ROOT_MOTION_REWARD

    return _ROOT_MOTION_PENALTY


def _stasis_score(previous_chord: Chord, chord: Chord) -> float:
    if previous_chord == chord:
        return -1.0

    return 0.0


def _repetition_variation_score(chords: tuple[Chord, ...]) -> float:
    longest_run = _longest_same_chord_run(chords)
    if longest_run <= _SAME_CHORD_RUN_GRACE_LENGTH:
        return 0.0

    return -float(longest_run - _SAME_CHORD_RUN_GRACE_LENGTH)


def _harmonic_diversity_score(chords: tuple[Chord, ...]) -> float:
    return len(set(chords)) / len(chords)


def _longest_same_chord_run(chords: tuple[Chord, ...]) -> int:
    longest_run = 0
    current_run = 0
    previous_chord: Chord | None = None
    for chord in chords:
        if chord == previous_chord:
            current_run += 1
        else:
            current_run = 1

        longest_run = max(longest_run, current_run)
        previous_chord = chord

    return longest_run


def _function_score(
    harmonic_function: HarmonicFunction | None,
    *,
    tonic: float,
    predominant: float,
    dominant: float,
) -> float:
    match harmonic_function:
        case HarmonicFunction.TONIC:
            return tonic
        case HarmonicFunction.PREDOMINANT:
            return predominant
        case HarmonicFunction.DOMINANT:
            return dominant
        case None:
            return 0.0


def _cadence_strength_for_role(role: HarmonicSlotRole) -> float:
    match role:
        case HarmonicSlotRole.OPENING:
            return _OPENING_CADENCE_STRENGTH
        case HarmonicSlotRole.CONTINUATION:
            return _CONTINUATION_CADENCE_STRENGTH
        case HarmonicSlotRole.CADENCE_PREPARATION:
            return _CADENCE_PREPARATION_STRENGTH
        case HarmonicSlotRole.CADENCE:
            return _CADENCE_STRENGTH


def _tension_level_for_role(role: HarmonicSlotRole) -> float:
    match role:
        case HarmonicSlotRole.OPENING | HarmonicSlotRole.CADENCE:
            return _LOW_TENSION
        case HarmonicSlotRole.CONTINUATION:
            return _CONTINUATION_TENSION
        case HarmonicSlotRole.CADENCE_PREPARATION:
            return _DOMINANT_TENSION


def _function_tension_level(harmonic_function: HarmonicFunction | None) -> float:
    match harmonic_function:
        case HarmonicFunction.TONIC:
            return _LOW_TENSION
        case HarmonicFunction.PREDOMINANT:
            return _PREDOMINANT_TENSION
        case HarmonicFunction.DOMINANT:
            return _DOMINANT_TENSION
        case None:
            return _CONTINUATION_TENSION


def _log_probability(probability: float) -> float:
    return log(max(probability, _LOG_PROBABILITY_FLOOR))
