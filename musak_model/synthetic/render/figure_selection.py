from collections.abc import Sequence
from fractions import Fraction

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from scipy.special import softmax

from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.synthetic.figures import FigureVocabularyEntry
from musak_model.synthetic.render.config import RenderConfig
from musak_model.synthetic.render.similarity import figure_edit_distance
from musak_model.synthetic.substitution.scoring import accent_fit, harmonic_fit, onset_chord_tone_fraction, slope_fit
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import ScaleType, scale_size_for_type


def figure_span_units(figure: FigureNGram) -> Fraction:
    return sum((duration for _, duration in figure.onsets), Fraction(0))


def slot_base_duration(figure: FigureNGram, slot_duration: Fraction) -> Fraction:
    return slot_duration / figure_span_units(figure)


def figure_fits_slot(
    figure: FigureNGram,
    *,
    slot_duration: Fraction,
    shortest_note_duration: Fraction,
    duration_vocabulary: DurationVocabulary,
) -> bool:
    base_duration = slot_base_duration(figure, slot_duration)
    for _, normalized_duration in figure.onsets:
        actual_duration = normalized_duration * base_duration
        if actual_duration < shortest_note_duration:
            return False

        if duration_vocabulary.duration_id_or_none(actual_duration) is None:
            return False

    return True


def select_figure(
    entries: Sequence[FigureVocabularyEntry],
    *,
    anchor: int,
    target_slope: int,
    scale_type: ScaleType,
    chord_pitch_classes: frozenset[int],
    weight: float,
    config: RenderConfig,
    rng: Generator,
    intended: FigureNGram | None = None,
    metrical_position: int | None = None,
    grid_count_per_bar: int | None = None,
) -> FigureVocabularyEntry:
    return select_scored_figure(
        entries,
        anchor=anchor,
        target_slope=target_slope,
        scale_type=scale_type,
        chord_pitch_classes=chord_pitch_classes,
        weight=weight,
        config=config,
        rng=rng,
        intended=intended,
        metrical_position=metrical_position,
        grid_count_per_bar=grid_count_per_bar,
    )[0]


def select_scored_figure(
    entries: Sequence[FigureVocabularyEntry],
    *,
    anchor: int,
    target_slope: int,
    scale_type: ScaleType,
    chord_pitch_classes: frozenset[int],
    weight: float,
    config: RenderConfig,
    rng: Generator,
    intended: FigureNGram | None = None,
    metrical_position: int | None = None,
    grid_count_per_bar: int | None = None,
) -> tuple[FigureVocabularyEntry, float]:
    if not entries:
        raise ValueError("entries must be non-empty")

    scores = figure_log_scores(
        entries,
        anchor=anchor,
        target_slope=target_slope,
        scale_type=scale_type,
        chord_pitch_classes=chord_pitch_classes,
        weight=weight,
        config=config,
        intended=intended,
        metrical_position=metrical_position,
        grid_count_per_bar=grid_count_per_bar,
    )
    index = int(rng.choice(len(entries), p=softmax(scores)))
    return entries[index], float(scores[index])


def figure_log_scores(
    entries: Sequence[FigureVocabularyEntry],
    *,
    anchor: int,
    target_slope: int,
    scale_type: ScaleType,
    chord_pitch_classes: frozenset[int],
    weight: float,
    config: RenderConfig,
    intended: FigureNGram | None = None,
    metrical_position: int | None = None,
    grid_count_per_bar: int | None = None,
) -> NDArray[np.float64]:
    counts = np.fromiter((entry.count for entry in entries), dtype=np.float64, count=len(entries))
    slopes = np.fromiter(
        (slope_fit(figure=entry.figure, target_slope=target_slope) for entry in entries),
        dtype=np.float64,
        count=len(entries),
    )
    if metrical_position is not None and grid_count_per_bar is not None:
        harmonic_values = [
            harmonic_fit(
                figure=entry.figure,
                anchor=anchor,
                scale_type=scale_type,
                chord_pitch_classes=chord_pitch_classes,
                metrical_position=metrical_position,
                grid_count_per_bar=grid_count_per_bar,
            )
            for entry in entries
        ]
    else:
        harmonic_values = [
            _chord_tone_coverage(
                entry.figure, anchor=anchor, scale_type=scale_type, chord_pitch_classes=chord_pitch_classes
            )
            for entry in entries
        ]
    harmonics = np.asarray(harmonic_values, dtype=np.float64)
    accents = np.fromiter(
        (
            accent_fit(
                figure=entry.figure,
                anchor=anchor,
                scale_type=scale_type,
                chord_pitch_classes=chord_pitch_classes,
                envelope_value=weight,
            )
            for entry in entries
        ),
        dtype=np.float64,
        count=len(entries),
    )
    scores = (
        config.commonness_bias * np.log(counts)
        + config.lambda_curve * slopes
        + config.lambda_harmonic * harmonics
        + config.lambda_accent * accents
    )
    if intended is not None and config.lambda_similarity != 0.0:
        similarities = np.fromiter(
            (-figure_edit_distance(entry.figure, intended) for entry in entries),
            dtype=np.float64,
            count=len(entries),
        )
        scores = scores + config.lambda_similarity * similarities

    return scores


def _chord_tone_coverage(
    figure: FigureNGram,
    *,
    anchor: int,
    scale_type: ScaleType,
    chord_pitch_classes: frozenset[int],
) -> float:
    scale_size = scale_size_for_type(scale_type)
    weighted = 0.0
    total = 0.0
    for degrees, normalized_duration in figure.onsets:
        duration_weight = float(normalized_duration)
        weighted += duration_weight * onset_chord_tone_fraction(
            degrees,
            anchor=anchor,
            scale_size=scale_size,
            scale_type=scale_type,
            chord_pitch_classes=chord_pitch_classes,
        )
        total += duration_weight

    if total == 0.0:
        return 0.0

    return weighted / total
