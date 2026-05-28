from collections.abc import Sequence

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from scipy.special import softmax

from musak_model.synthetic.figures import FigureVocabulary, FigureVocabularyEntry
from musak_model.synthetic.substitution.config import SubstitutionConfig
from musak_model.synthetic.substitution.scoring import harm_fit, is_monorhythmic, slope_fit
from musak_model.tokens.schema import Hand, ScaleType


def monorhythmic_entries(
    vocabulary: FigureVocabulary,
    *,
    scale_type: ScaleType,
    hand: Hand,
    figure_length: int,
) -> tuple[FigureVocabularyEntry, ...]:
    filtered = vocabulary.filter(scale_type=scale_type, hand=hand, n=figure_length).entries
    return tuple(entry for entry in filtered if is_monorhythmic(entry.figure))


def tilted_log_probabilities(
    *,
    entries: Sequence[FigureVocabularyEntry],
    anchor: int,
    target_slope: int,
    scale_type: ScaleType,
    chord_pitch_classes: frozenset[int],
    config: SubstitutionConfig,
) -> NDArray[np.float64]:
    counts = np.fromiter((entry.count for entry in entries), dtype=np.float64, count=len(entries))
    log_p_emp = config.commonness_bias * np.log(counts)
    slope_scores = np.fromiter(
        (slope_fit(figure=entry.figure, target_slope=target_slope) for entry in entries),
        dtype=np.float64,
        count=len(entries),
    )
    harm_scores = np.fromiter(
        (
            harm_fit(
                figure=entry.figure,
                anchor=anchor,
                scale_type=scale_type,
                chord_pitch_classes=chord_pitch_classes,
            )
            for entry in entries
        ),
        dtype=np.float64,
        count=len(entries),
    )
    return _combine_log_terms(
        log_p_emp=log_p_emp,
        slope_scores=slope_scores,
        harm_scores=harm_scores,
        lambda_curve=config.lambda_curve,
        lambda_harm=config.lambda_harm,
    )


def sample_substituted_figure(
    *,
    entries: Sequence[FigureVocabularyEntry],
    anchor: int,
    target_slope: int,
    scale_type: ScaleType,
    chord_pitch_classes: frozenset[int],
    config: SubstitutionConfig,
    rng: Generator,
) -> FigureVocabularyEntry:
    if not entries:
        raise ValueError("entries must be non-empty")

    log_probabilities = tilted_log_probabilities(
        entries=entries,
        anchor=anchor,
        target_slope=target_slope,
        scale_type=scale_type,
        chord_pitch_classes=chord_pitch_classes,
        config=config,
    )
    probabilities = softmax(log_probabilities)
    return entries[int(rng.choice(len(entries), p=probabilities))]


def _combine_log_terms(
    *,
    log_p_emp: NDArray[np.float64],
    slope_scores: NDArray[np.float64],
    harm_scores: NDArray[np.float64],
    lambda_curve: float,
    lambda_harm: float,
) -> NDArray[np.float64]:
    return log_p_emp + lambda_curve * slope_scores + lambda_harm * harm_scores
