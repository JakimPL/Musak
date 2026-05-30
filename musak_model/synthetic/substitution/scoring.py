from fractions import Fraction
from math import gcd

from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.tokens.pitch import degree_pitch_class, diatonic_position_to_degree_and_octave
from musak_model.tokens.schema import ScaleType, scale_size_for_type


def is_monorhythmic(figure: FigureNGram) -> bool:
    return all(duration == Fraction(1) for _, duration in figure.onsets)


def figure_net_contour(figure: FigureNGram) -> int:
    return min(position for position, _ in figure.onsets[-1][0])


def slope_fit(*, figure: FigureNGram, target_slope: int) -> float:
    return float(-abs(figure_net_contour(figure) - target_slope))


def harm_fit(
    *,
    figure: FigureNGram,
    anchor: int,
    scale_type: ScaleType,
    chord_pitch_classes: frozenset[int],
    metrical_position: int,
    grid_count_per_bar: int,
) -> float:
    scale_size = scale_size_for_type(scale_type)
    weighted_chord_tone = 0.0
    weight_total = 0.0
    for onset_index, (degrees, _) in enumerate(figure.onsets):
        position = (metrical_position + onset_index) % grid_count_per_bar
        weight = gcd(position, grid_count_per_bar) / grid_count_per_bar
        weighted_chord_tone += weight * _onset_chord_tone_fraction(
            degrees,
            anchor=anchor,
            scale_size=scale_size,
            scale_type=scale_type,
            chord_pitch_classes=chord_pitch_classes,
        )
        weight_total += weight

    if weight_total == 0.0:
        return 0.0

    return weighted_chord_tone / weight_total


def accent_fit(
    *,
    figure: FigureNGram,
    anchor: int,
    scale_type: ScaleType,
    chord_pitch_classes: frozenset[int],
    envelope_value: float,
) -> float:
    durations = [float(duration) for _, duration in figure.onsets]
    total = sum(durations)
    if total <= 0.0:
        return 0.0

    scale_size = scale_size_for_type(scale_type)
    onset_count = len(durations)
    duration_weights = [duration / total for duration in durations]
    chord_tone_fractions = [
        _onset_chord_tone_fraction(
            degrees,
            anchor=anchor,
            scale_size=scale_size,
            scale_type=scale_type,
            chord_pitch_classes=chord_pitch_classes,
        )
        for degrees, _ in figure.onsets
    ]
    chord_total = sum(chord_tone_fractions)
    chord_weights = (
        [fraction / chord_total for fraction in chord_tone_fractions] if chord_total > 0.0 else duration_weights
    )
    stress = sum(
        (gcd(position, onset_count) / onset_count) * 0.5 * (duration_weights[position] + chord_weights[position])
        for position in range(onset_count)
    )

    return stress * envelope_value


def _onset_chord_tone_fraction(
    degrees: tuple[tuple[int, int], ...],
    *,
    anchor: int,
    scale_size: int,
    scale_type: ScaleType,
    chord_pitch_classes: frozenset[int],
) -> float:
    if not degrees:
        return 0.0

    chord_tones = sum(
        degree_pitch_class(
            diatonic_position_to_degree_and_octave(anchor + relative_position, scale_size=scale_size)[0],
            accidental,
            scale_type=scale_type,
        )
        in chord_pitch_classes
        for relative_position, accidental in degrees
    )
    return chord_tones / len(degrees)
