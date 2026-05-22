from fractions import Fraction

from musak_model.data.scale_matcher.candidates import (
    candidate_explanations,
    matches_declared_pitch_set,
    ranked_candidates,
    second_score,
    selected_explanation,
    tied_best_candidate_count,
)
from musak_model.data.scale_matcher.config import ScaleMatcherConfig
from musak_model.data.scale_matcher.histogram import declared_key_fifths, normalized_histogram, pitch_class_histogram
from musak_model.data.scale_matcher.schema import ScaleMatch
from musak_model.data.schema import ParsedBar, ScaleMatchDiagnostics
from musak_shared.elements import pitch_class_from_key_fifths


def match_scale(
    right_hand_bars: list[ParsedBar],
    left_hand_bars: list[ParsedBar],
    *,
    config: ScaleMatcherConfig,
) -> ScaleMatch:
    bars = right_hand_bars + left_hand_bars
    histogram = pitch_class_histogram(bars)
    return match_scale_histogram(
        histogram,
        declared_key_fifths=declared_key_fifths(bars),
        config=config,
    )


def match_scale_histogram(
    histogram: dict[int, Fraction],
    *,
    declared_key_fifths: int | None,
    config: ScaleMatcherConfig,
) -> ScaleMatch:
    histogram = normalized_histogram(histogram)
    declared_scale_root = pitch_class_from_key_fifths(declared_key_fifths) if declared_key_fifths is not None else None
    candidates = ranked_candidates(histogram)
    explanations = candidate_explanations(
        candidates,
        histogram=histogram,
        support_score_margin=config.support_score_margin,
    )
    selected_scale_explanation = selected_explanation(
        explanations,
        declared_scale_root=declared_scale_root,
        selection_score_margin=config.selection_score_margin,
    )
    selected = selected_scale_explanation.candidate
    strict_best_score = candidates[0].in_scale_weight_fraction
    selected_second_score = second_score(candidates, best_score=strict_best_score)
    tied_best_count = tied_best_candidate_count(candidates, best_score=strict_best_score)
    declared_match_used = matches_declared_pitch_set(selected, declared_scale_root=declared_scale_root)
    no_pitches = sum(histogram.values(), Fraction(0)) == 0
    ambiguous = tied_best_count > 1 and not declared_match_used
    best_margin = max(0.0, strict_best_score - selected_second_score)
    low_confidence = (
        no_pitches
        or selected_scale_explanation.unexplained_out_of_scale_weight_fraction
        > config.maximum_unexplained_weight_fraction
        or selected_scale_explanation.explanation_pitch_class_count > config.maximum_explanation_pitch_class_count
    )

    return ScaleMatch(
        scale_root=selected.scale_root,
        scale_type=selected.scale_type,
        diagnostics=ScaleMatchDiagnostics(
            declared_key_fifths=declared_key_fifths,
            in_scale_weight_fraction=selected.in_scale_weight_fraction,
            out_of_scale_weight_fraction=1.0 - selected.in_scale_weight_fraction,
            explained_out_of_scale_weight_fraction=selected_scale_explanation.explained_out_of_scale_weight_fraction,
            unexplained_out_of_scale_weight_fraction=(
                selected_scale_explanation.unexplained_out_of_scale_weight_fraction
            ),
            best_margin=best_margin,
            observed_pitch_class_count=sum(1 for weight in histogram.values() if weight > 0),
            explanation_pitch_class_count=selected_scale_explanation.explanation_pitch_class_count,
            support_candidate_count=selected_scale_explanation.support_candidate_count,
            tied_best_candidate_count=tied_best_count,
            declared_match_used=declared_match_used,
            low_confidence=low_confidence,
            ambiguous=ambiguous,
            no_pitches=no_pitches,
        ),
    )
