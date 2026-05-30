from collections.abc import Mapping, Sequence

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from scipy.special import softmax

from musak_model.harmony.schema import Chord
from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.synthetic.figures import FigureVocabulary, FigureVocabularyEntry
from musak_model.synthetic.substitution.chord_figure import FigureByChordModel
from musak_model.synthetic.substitution.config import SubstitutionConfig
from musak_model.synthetic.substitution.scoring import (
    accent_fit,
    chord_figure_log_probabilities,
    harm_fit,
    is_monorhythmic,
    slope_fit,
)
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
    envelope_value: float,
    metrical_position: int,
    grid_count_per_bar: int,
    config: SubstitutionConfig,
    chord: Chord | None = None,
    figure_by_chord_model: FigureByChordModel = FigureByChordModel(),
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
                metrical_position=metrical_position,
                grid_count_per_bar=grid_count_per_bar,
            )
            for entry in entries
        ),
        dtype=np.float64,
        count=len(entries),
    )
    accent_scores = np.fromiter(
        (
            accent_fit(
                figure=entry.figure,
                anchor=anchor,
                scale_type=scale_type,
                chord_pitch_classes=chord_pitch_classes,
                envelope_value=envelope_value,
            )
            for entry in entries
        ),
        dtype=np.float64,
        count=len(entries),
    )
    chord_figure_scores = np.asarray(
        chord_figure_log_probabilities(
            figures=[entry.figure for entry in entries],
            table=_chord_figure_table(entries, chord=chord, scale_type=scale_type, model=figure_by_chord_model),
        ),
        dtype=np.float64,
    )
    return _combine_log_terms(
        log_p_emp=log_p_emp,
        slope_scores=slope_scores,
        harm_scores=harm_scores,
        accent_scores=accent_scores,
        chord_figure_scores=chord_figure_scores,
        lambda_curve=config.lambda_curve,
        lambda_harm=config.lambda_harm,
        lambda_accent=config.lambda_accent,
        lambda_chord_figure=config.lambda_chord_figure,
    )


def _chord_figure_table(
    entries: Sequence[FigureVocabularyEntry],
    *,
    chord: Chord | None,
    scale_type: ScaleType,
    model: FigureByChordModel,
) -> Mapping[FigureNGram, float] | None:
    if not entries:
        return None

    group = entries[0].group
    return model.table(scale_type=scale_type, hand=group.hand, figure_length=group.n, chord=chord)


def sample_substituted_figure(
    *,
    entries: Sequence[FigureVocabularyEntry],
    anchor: int,
    target_slope: int,
    scale_type: ScaleType,
    chord_pitch_classes: frozenset[int],
    envelope_value: float,
    metrical_position: int,
    grid_count_per_bar: int,
    config: SubstitutionConfig,
    rng: Generator,
    chord: Chord | None = None,
    figure_by_chord_model: FigureByChordModel = FigureByChordModel(),
) -> FigureVocabularyEntry:
    if not entries:
        raise ValueError("entries must be non-empty")

    log_probabilities = tilted_log_probabilities(
        entries=entries,
        anchor=anchor,
        target_slope=target_slope,
        scale_type=scale_type,
        chord_pitch_classes=chord_pitch_classes,
        envelope_value=envelope_value,
        metrical_position=metrical_position,
        grid_count_per_bar=grid_count_per_bar,
        config=config,
        chord=chord,
        figure_by_chord_model=figure_by_chord_model,
    )
    probabilities = softmax(log_probabilities)
    return entries[int(rng.choice(len(entries), p=probabilities))]


def _combine_log_terms(
    *,
    log_p_emp: NDArray[np.float64],
    slope_scores: NDArray[np.float64],
    harm_scores: NDArray[np.float64],
    accent_scores: NDArray[np.float64],
    chord_figure_scores: NDArray[np.float64],
    lambda_curve: float,
    lambda_harm: float,
    lambda_accent: float,
    lambda_chord_figure: float,
) -> NDArray[np.float64]:
    return (
        log_p_emp
        + lambda_curve * slope_scores
        + lambda_harm * harm_scores
        + lambda_accent * accent_scores
        + lambda_chord_figure * chord_figure_scores
    )
