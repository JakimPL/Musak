from fractions import Fraction

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
) -> float:
    scale_size = scale_size_for_type(scale_type)
    chord_tone_count = 0
    total_count = 0
    for degrees, _ in figure.onsets:
        for relative_position, accidental in degrees:
            absolute_position = anchor + relative_position
            absolute_degree, _octave_offset = diatonic_position_to_degree_and_octave(
                absolute_position, scale_size=scale_size
            )
            note_pitch_class = degree_pitch_class(absolute_degree, accidental, scale_type=scale_type)
            if note_pitch_class in chord_pitch_classes:
                chord_tone_count += 1

            total_count += 1

    if total_count == 0:
        return 0.0

    return chord_tone_count / total_count
