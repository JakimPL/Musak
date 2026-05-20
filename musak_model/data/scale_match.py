from dataclasses import dataclass
from fractions import Fraction
from typing import Final

from musak_model.data.schema import ParsedBar, ParsedChord, ParsedNote, ScaleMatchDiagnostics
from musak_model.tokens.schema import SCALE_INTERVALS, ScaleType
from musak_shared.elements import PITCHES_PER_OCTAVE

_SCORE_EPSILON: Final[float] = 1e-12
_CANDIDATE_SCALE_TYPES: Final[tuple[ScaleType, ...]] = (
    ScaleType.MAJOR,
    ScaleType.HARMONIC_MINOR,
    ScaleType.MELODIC_MINOR,
)


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


def match_scale(
    right_hand_bars: list[ParsedBar],
    left_hand_bars: list[ParsedBar],
    *,
    minimum_in_scale_weight_fraction: float,
    minimum_best_margin: float,
) -> ScaleMatch:
    histogram = _pitch_class_histogram(right_hand_bars + left_hand_bars)
    declared_key_fifths = _declared_key_fifths(right_hand_bars + left_hand_bars)
    declared_scale_root = _declared_scale_root(declared_key_fifths)
    candidates = _ranked_candidates(histogram)
    selected = _selected_candidate(candidates, declared_scale_root=declared_scale_root)
    best_score = selected.in_scale_weight_fraction
    second_score = _second_score(candidates, best_score=best_score)
    tied_best_candidate_count = _tied_best_candidate_count(candidates, best_score=best_score)
    declared_match_used = _matches_declared_pitch_set(selected, declared_scale_root=declared_scale_root) and any(
        _matches_declared_pitch_set(candidate, declared_scale_root=declared_scale_root)
        and abs(candidate.in_scale_weight_fraction - best_score) <= _SCORE_EPSILON
        for candidate in candidates
    )
    no_pitches = sum(histogram.values(), Fraction(0)) == 0
    ambiguous = tied_best_candidate_count > 1 and not declared_match_used
    best_margin = max(0.0, best_score - second_score)
    low_confidence = (
        no_pitches
        or best_score < minimum_in_scale_weight_fraction
        or (best_margin < minimum_best_margin and not declared_match_used)
    )

    return ScaleMatch(
        scale_root=selected.scale_root,
        scale_type=selected.scale_type,
        diagnostics=ScaleMatchDiagnostics(
            declared_key_fifths=declared_key_fifths,
            declared_scale_root=declared_scale_root,
            in_scale_weight_fraction=best_score,
            out_of_scale_weight_fraction=1.0 - best_score,
            best_margin=best_margin,
            observed_pitch_class_count=sum(1 for weight in histogram.values() if weight > 0),
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


def _declared_key_fifths(bars: list[ParsedBar]) -> int | None:
    for bar in bars:
        if bar.declared_key_fifths is not None:
            return bar.declared_key_fifths

    return None


def _declared_scale_root(declared_key_fifths: int | None) -> int | None:
    if declared_key_fifths is None:
        return None

    return (declared_key_fifths * 7) % PITCHES_PER_OCTAVE


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
        return _ScaleCandidate(scale_root=scale_root, scale_type=scale_type, in_scale_weight_fraction=0.0)

    scale_pitch_classes = {(scale_root + interval) % PITCHES_PER_OCTAVE for interval in SCALE_INTERVALS[scale_type]}
    in_scale_weight = sum(histogram[pitch_class] for pitch_class in scale_pitch_classes)
    return _ScaleCandidate(
        scale_root=scale_root,
        scale_type=scale_type,
        in_scale_weight_fraction=float(in_scale_weight / total_weight),
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
    return min(abs(fifths) for fifths in range(-6, 7) if (fifths * 7) % PITCHES_PER_OCTAVE == scale_root)


def _selected_candidate(
    candidates: list[_ScaleCandidate],
    *,
    declared_scale_root: int | None,
) -> _ScaleCandidate:
    best_score = candidates[0].in_scale_weight_fraction
    tied_candidates = [
        candidate for candidate in candidates if abs(candidate.in_scale_weight_fraction - best_score) <= _SCORE_EPSILON
    ]
    if declared_scale_root is not None:
        for candidate in tied_candidates:
            if _matches_declared_pitch_set(candidate, declared_scale_root=declared_scale_root):
                return candidate

    return tied_candidates[0]


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
