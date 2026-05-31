from collections.abc import Sequence
from fractions import Fraction

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from scipy.special import softmax

from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.synthetic.figures import FigureVocabularyEntry
from musak_model.synthetic.render.config import RenderConfig
from musak_model.synthetic.substitution.scoring import accent_fit, onset_chord_tone_fraction, slope_fit
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
) -> FigureVocabularyEntry:
    if not entries:
        raise ValueError("entries must be non-empty")

    probabilities = softmax(
        _tilted_log_probabilities(
            entries,
            anchor=anchor,
            target_slope=target_slope,
            scale_type=scale_type,
            chord_pitch_classes=chord_pitch_classes,
            weight=weight,
            config=config,
        )
    )
    return entries[int(rng.choice(len(entries), p=probabilities))]


def _tilted_log_probabilities(
    entries: Sequence[FigureVocabularyEntry],
    *,
    anchor: int,
    target_slope: int,
    scale_type: ScaleType,
    chord_pitch_classes: frozenset[int],
    weight: float,
    config: RenderConfig,
) -> NDArray[np.float64]:
    counts = np.fromiter((entry.count for entry in entries), dtype=np.float64, count=len(entries))
    slopes = np.fromiter(
        (slope_fit(figure=entry.figure, target_slope=target_slope) for entry in entries),
        dtype=np.float64,
        count=len(entries),
    )
    harmonics = np.fromiter(
        (
            _chord_tone_coverage(
                entry.figure, anchor=anchor, scale_type=scale_type, chord_pitch_classes=chord_pitch_classes
            )
            for entry in entries
        ),
        dtype=np.float64,
        count=len(entries),
    )
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
    return (
        config.commonness_bias * np.log(counts)
        + config.lambda_curve * slopes
        + config.lambda_harmonic * harmonics
        + config.lambda_accent * accents
    )


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
