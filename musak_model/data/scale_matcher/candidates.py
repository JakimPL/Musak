from fractions import Fraction
from typing import Final

from musak_model.data.scale_matcher.schema import CandidateExplanation, ScaleCandidate
from musak_model.tokens.schema import SCALE_INTERVALS, ScaleType
from musak_shared.elements import PITCHES_PER_OCTAVE, key_fifths_from_pitch_class

_MIN_SUPPORT_PITCH_CLASS_OVERLAP: Final[int] = 5
_SCORE_EPSILON: Final[float] = 1e-12


def ranked_candidates(histogram: dict[int, Fraction]) -> list[ScaleCandidate]:
    total_weight = sum(histogram.values(), Fraction(0))
    candidates = [
        candidate(
            scale_root=scale_root,
            scale_type=scale_type,
            histogram=histogram,
            total_weight=total_weight,
        )
        for scale_root in range(PITCHES_PER_OCTAVE)
        for scale_type in ScaleType
    ]
    return sorted(candidates, key=candidate_sort_key)


def candidate(
    *,
    scale_root: int,
    scale_type: ScaleType,
    histogram: dict[int, Fraction],
    total_weight: Fraction,
) -> ScaleCandidate:
    if total_weight == 0:
        return ScaleCandidate(
            scale_root=scale_root,
            scale_type=scale_type,
            in_scale_weight_fraction=0.0,
            pitch_classes=frozenset(),
        )

    scale_pitch_classes = {(scale_root + interval) % PITCHES_PER_OCTAVE for interval in SCALE_INTERVALS[scale_type]}
    in_scale_weight = sum(histogram[pitch_class] for pitch_class in scale_pitch_classes)
    return ScaleCandidate(
        scale_root=scale_root,
        scale_type=scale_type,
        in_scale_weight_fraction=float(in_scale_weight / total_weight),
        pitch_classes=frozenset(scale_pitch_classes),
    )


def candidate_sort_key(candidate: ScaleCandidate) -> tuple[float, int, int, int]:
    return (
        -candidate.in_scale_weight_fraction,
        scale_type_sort_index(candidate.scale_type),
        root_complexity(candidate.scale_root),
        candidate.scale_root,
    )


def scale_type_sort_index(scale_type: ScaleType) -> int:
    return list(ScaleType).index(scale_type)


def root_complexity(scale_root: int) -> int:
    return abs(key_fifths_from_pitch_class(scale_root))


def candidate_explanations(
    candidates: list[ScaleCandidate],
    *,
    histogram: dict[int, Fraction],
    support_score_margin: float,
) -> list[CandidateExplanation]:
    return [
        candidate_explanation(
            candidate,
            candidates=candidates,
            histogram=histogram,
            support_score_margin=support_score_margin,
        )
        for candidate in candidates
    ]


def candidate_explanation(
    candidate: ScaleCandidate,
    *,
    candidates: list[ScaleCandidate],
    histogram: dict[int, Fraction],
    support_score_margin: float,
) -> CandidateExplanation:
    total_weight = sum(histogram.values(), Fraction(0))
    support_candidates = scale_support_candidates(
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

    return CandidateExplanation(
        candidate=candidate,
        explained_out_of_scale_weight_fraction=explained_out_of_scale_weight_fraction,
        unexplained_out_of_scale_weight_fraction=unexplained_out_of_scale_weight_fraction,
        explanation_pitch_class_count=len(observed_explanation_pitch_classes),
        support_candidate_count=len(support_candidates),
    )


def scale_support_candidates(
    candidate: ScaleCandidate,
    *,
    candidates: list[ScaleCandidate],
    support_score_margin: float,
) -> list[ScaleCandidate]:
    return [
        support
        for support in candidates
        if candidate.in_scale_weight_fraction - support.in_scale_weight_fraction <= support_score_margin
        and len(candidate.pitch_classes & support.pitch_classes) >= _MIN_SUPPORT_PITCH_CLASS_OVERLAP
    ]


def selected_explanation(
    explanations: list[CandidateExplanation],
    *,
    declared_scale_root: int | None,
    selection_score_margin: float,
) -> CandidateExplanation:
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
            not matches_declared_pitch_set(explanation.candidate, declared_scale_root=declared_scale_root),
            candidate_sort_key(explanation.candidate),
        ),
    )


def matches_declared_pitch_set(candidate: ScaleCandidate, *, declared_scale_root: int | None) -> bool:
    return (
        declared_scale_root is not None
        and candidate.scale_root == declared_scale_root
        and candidate.scale_type == ScaleType.MAJOR
    )


def second_score(candidates: list[ScaleCandidate], *, best_score: float) -> float:
    for candidate in candidates:
        if abs(candidate.in_scale_weight_fraction - best_score) > _SCORE_EPSILON:
            return candidate.in_scale_weight_fraction

    return best_score


def tied_best_candidate_count(candidates: list[ScaleCandidate], *, best_score: float) -> int:
    return sum(1 for candidate in candidates if abs(candidate.in_scale_weight_fraction - best_score) <= _SCORE_EPSILON)
