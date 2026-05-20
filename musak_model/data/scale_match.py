from dataclasses import dataclass
from fractions import Fraction
from typing import Final

from musak_model.data.schema import ParsedBar, ParsedChord, ParsedNote, ScaleMatchDiagnostics
from musak_model.tokens.schema import SCALE_INTERVALS, ScaleType
from musak_shared.elements import PITCHES_PER_OCTAVE, key_fifths_from_pitch_class, pitch_class_from_key_fifths

_SCORE_EPSILON: Final[float] = 1e-12
_CANDIDATE_SCALE_TYPES: Final[tuple[ScaleType, ...]] = (
    ScaleType.MAJOR,
    ScaleType.HARMONIC_MINOR,
    ScaleType.MELODIC_MINOR,
)
_MIN_SUPPORT_PITCH_CLASS_OVERLAP: Final[int] = 5


@dataclass(frozen=True)
class ScaleMatch:
    scale_root: int
    scale_type: ScaleType
    diagnostics: ScaleMatchDiagnostics


@dataclass(frozen=True)
class _ScaleCandidate:
    scale_root: int
    scale_type: ScaleType
    in_scale_weight_fraction: float
    pitch_classes: frozenset[int]


@dataclass(frozen=True)
class _CandidateExplanation:
    candidate: _ScaleCandidate
    explained_out_of_scale_weight_fraction: float
    unexplained_out_of_scale_weight_fraction: float
    explanation_pitch_class_count: int
    support_candidate_count: int


def match_scale(
    right_hand_bars: list[ParsedBar],
    left_hand_bars: list[ParsedBar],
    *,
    support_score_margin: float,
    selection_score_margin: float,
    maximum_unexplained_weight_fraction: float,
    maximum_explanation_pitch_class_count: int,
) -> ScaleMatch:
    histogram = _pitch_class_histogram(right_hand_bars + left_hand_bars)
    declared_key_fifths = _declared_key_fifths(right_hand_bars + left_hand_bars)
    return match_scale_histogram(
        histogram,
        declared_key_fifths=declared_key_fifths,
        support_score_margin=support_score_margin,
        selection_score_margin=selection_score_margin,
        maximum_unexplained_weight_fraction=maximum_unexplained_weight_fraction,
        maximum_explanation_pitch_class_count=maximum_explanation_pitch_class_count,
    )


def match_scale_histogram(
    histogram: dict[int, Fraction],
    *,
    declared_key_fifths: int | None,
    support_score_margin: float,
    selection_score_margin: float,
    maximum_unexplained_weight_fraction: float,
    maximum_explanation_pitch_class_count: int,
) -> ScaleMatch:
    histogram = _normalized_histogram(histogram)
    declared_scale_root = pitch_class_from_key_fifths(declared_key_fifths) if declared_key_fifths is not None else None
    candidates = _ranked_candidates(histogram)
    explanations = _candidate_explanations(
        candidates,
        histogram=histogram,
        support_score_margin=support_score_margin,
    )
    selected_explanation = _selected_explanation(
        explanations,
        declared_scale_root=declared_scale_root,
        selection_score_margin=selection_score_margin,
    )
    selected = selected_explanation.candidate
    strict_best_score = candidates[0].in_scale_weight_fraction
    second_score = _second_score(candidates, best_score=strict_best_score)
    tied_best_candidate_count = _tied_best_candidate_count(candidates, best_score=strict_best_score)
    declared_match_used = _matches_declared_pitch_set(selected, declared_scale_root=declared_scale_root)
    no_pitches = sum(histogram.values(), Fraction(0)) == 0
    ambiguous = tied_best_candidate_count > 1 and not declared_match_used
    best_margin = max(0.0, strict_best_score - second_score)
    low_confidence = (
        no_pitches
        or selected_explanation.unexplained_out_of_scale_weight_fraction > maximum_unexplained_weight_fraction
        or selected_explanation.explanation_pitch_class_count > maximum_explanation_pitch_class_count
    )

    return ScaleMatch(
        scale_root=selected.scale_root,
        scale_type=selected.scale_type,
        diagnostics=ScaleMatchDiagnostics(
            declared_key_fifths=declared_key_fifths,
            in_scale_weight_fraction=selected.in_scale_weight_fraction,
            out_of_scale_weight_fraction=1.0 - selected.in_scale_weight_fraction,
            explained_out_of_scale_weight_fraction=selected_explanation.explained_out_of_scale_weight_fraction,
            unexplained_out_of_scale_weight_fraction=selected_explanation.unexplained_out_of_scale_weight_fraction,
            best_margin=best_margin,
            observed_pitch_class_count=sum(1 for weight in histogram.values() if weight > 0),
            explanation_pitch_class_count=selected_explanation.explanation_pitch_class_count,
            support_candidate_count=selected_explanation.support_candidate_count,
            tied_best_candidate_count=tied_best_candidate_count,
            declared_match_used=declared_match_used,
            low_confidence=low_confidence,
            ambiguous=ambiguous,
            no_pitches=no_pitches,
        ),
    )


def _pitch_class_histogram(bars: list[ParsedBar]) -> dict[int, Fraction]:
    histogram = {pitch_class: Fraction(0) for pitch_class in range(PITCHES_PER_OCTAVE)}
    for bar in bars:
        for event in bar.events:
            match event:
                case ParsedNote():
                    histogram[event.midi_pitch % PITCHES_PER_OCTAVE] += event.duration
                case ParsedChord():
                    for midi_pitch in event.midi_pitches:
                        histogram[midi_pitch % PITCHES_PER_OCTAVE] += event.duration
    return histogram


def _normalized_histogram(histogram: dict[int, Fraction]) -> dict[int, Fraction]:
    normalized = {pitch_class: Fraction(0) for pitch_class in range(PITCHES_PER_OCTAVE)}
    for pitch_class, weight in histogram.items():
        if pitch_class < 0 or pitch_class >= PITCHES_PER_OCTAVE:
            raise ValueError(f"pitch class must be in [0, {PITCHES_PER_OCTAVE - 1}], got {pitch_class}")
        if weight < 0:
            raise ValueError("pitch-class weights must be non-negative")

        normalized[pitch_class] += weight

    return normalized


def _declared_key_fifths(bars: list[ParsedBar]) -> int | None:
    for bar in bars:
        if bar.declared_key_fifths is not None:
            return bar.declared_key_fifths

    return None


def _ranked_candidates(histogram: dict[int, Fraction]) -> list[_ScaleCandidate]:
    total_weight = sum(histogram.values(), Fraction(0))
    candidates = [
        _candidate(
            scale_root=scale_root,
            scale_type=scale_type,
            histogram=histogram,
            total_weight=total_weight,
        )
        for scale_root in range(PITCHES_PER_OCTAVE)
        for scale_type in _CANDIDATE_SCALE_TYPES
    ]
    return sorted(candidates, key=_candidate_sort_key)


def _candidate(
    *,
    scale_root: int,
    scale_type: ScaleType,
    histogram: dict[int, Fraction],
    total_weight: Fraction,
) -> _ScaleCandidate:
    if total_weight == 0:
        return _ScaleCandidate(
            scale_root=scale_root,
            scale_type=scale_type,
            in_scale_weight_fraction=0.0,
            pitch_classes=frozenset(),
        )

    scale_pitch_classes = {(scale_root + interval) % PITCHES_PER_OCTAVE for interval in SCALE_INTERVALS[scale_type]}
    in_scale_weight = sum(histogram[pitch_class] for pitch_class in scale_pitch_classes)
    return _ScaleCandidate(
        scale_root=scale_root,
        scale_type=scale_type,
        in_scale_weight_fraction=float(in_scale_weight / total_weight),
        pitch_classes=frozenset(scale_pitch_classes),
    )


def _candidate_sort_key(candidate: _ScaleCandidate) -> tuple[float, int, int, int]:
    return (
        -candidate.in_scale_weight_fraction,
        _scale_type_sort_index(candidate.scale_type),
        _root_complexity(candidate.scale_root),
        candidate.scale_root,
    )


def _scale_type_sort_index(scale_type: ScaleType) -> int:
    return _CANDIDATE_SCALE_TYPES.index(scale_type)


def _root_complexity(scale_root: int) -> int:
    return abs(key_fifths_from_pitch_class(scale_root))


def _candidate_explanations(
    candidates: list[_ScaleCandidate],
    *,
    histogram: dict[int, Fraction],
    support_score_margin: float,
) -> list[_CandidateExplanation]:
    return [
        _candidate_explanation(
            candidate,
            candidates=candidates,
            histogram=histogram,
            support_score_margin=support_score_margin,
        )
        for candidate in candidates
    ]


def _candidate_explanation(
    candidate: _ScaleCandidate,
    *,
    candidates: list[_ScaleCandidate],
    histogram: dict[int, Fraction],
    support_score_margin: float,
) -> _CandidateExplanation:
    total_weight = sum(histogram.values(), Fraction(0))
    support_candidates = _support_candidates(
        candidate, candidates=candidates, support_score_margin=support_score_margin
    )
    observed_pitch_classes = frozenset(pitch_class for pitch_class, weight in histogram.items() if weight > 0)
    explanation_pitch_classes = frozenset().union(*(support.pitch_classes for support in support_candidates))
    observed_explanation_pitch_classes = explanation_pitch_classes & observed_pitch_classes
    if total_weight == 0:
        explained_out_of_scale_weight_fraction = 0.0
        unexplained_out_of_scale_weight_fraction = 1.0
    else:
        explained_out_of_scale_weight = sum(
            weight
            for pitch_class, weight in histogram.items()
            if pitch_class not in candidate.pitch_classes and pitch_class in observed_explanation_pitch_classes
        )
        unexplained_out_of_scale_weight = sum(
            weight for pitch_class, weight in histogram.items() if pitch_class not in observed_explanation_pitch_classes
        )
        explained_out_of_scale_weight_fraction = float(explained_out_of_scale_weight / total_weight)
        unexplained_out_of_scale_weight_fraction = float(unexplained_out_of_scale_weight / total_weight)

    return _CandidateExplanation(
        candidate=candidate,
        explained_out_of_scale_weight_fraction=explained_out_of_scale_weight_fraction,
        unexplained_out_of_scale_weight_fraction=unexplained_out_of_scale_weight_fraction,
        explanation_pitch_class_count=len(observed_explanation_pitch_classes),
        support_candidate_count=len(support_candidates),
    )


def _support_candidates(
    candidate: _ScaleCandidate,
    *,
    candidates: list[_ScaleCandidate],
    support_score_margin: float,
) -> list[_ScaleCandidate]:
    return [
        support
        for support in candidates
        if candidate.in_scale_weight_fraction - support.in_scale_weight_fraction <= support_score_margin
        and len(candidate.pitch_classes & support.pitch_classes) >= _MIN_SUPPORT_PITCH_CLASS_OVERLAP
    ]


def _selected_explanation(
    explanations: list[_CandidateExplanation],
    *,
    declared_scale_root: int | None,
    selection_score_margin: float,
) -> _CandidateExplanation:
    best_score = explanations[0].candidate.in_scale_weight_fraction
    close_explanations = [
        explanation
        for explanation in explanations
        if best_score - explanation.candidate.in_scale_weight_fraction <= selection_score_margin
    ]
    return min(
        close_explanations,
        key=lambda explanation: (
            explanation.unexplained_out_of_scale_weight_fraction,
            -explanation.candidate.in_scale_weight_fraction,
            not _matches_declared_pitch_set(explanation.candidate, declared_scale_root=declared_scale_root),
            _candidate_sort_key(explanation.candidate),
        ),
    )


def _matches_declared_pitch_set(candidate: _ScaleCandidate, *, declared_scale_root: int | None) -> bool:
    return (
        declared_scale_root is not None
        and candidate.scale_root == declared_scale_root
        and candidate.scale_type == ScaleType.MAJOR
    )


def _second_score(candidates: list[_ScaleCandidate], *, best_score: float) -> float:
    for candidate in candidates:
        if abs(candidate.in_scale_weight_fraction - best_score) > _SCORE_EPSILON:
            return candidate.in_scale_weight_fraction

    return best_score


def _tied_best_candidate_count(candidates: list[_ScaleCandidate], *, best_score: float) -> int:
    return sum(1 for candidate in candidates if abs(candidate.in_scale_weight_fraction - best_score) <= _SCORE_EPSILON)
