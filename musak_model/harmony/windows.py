from fractions import Fraction


def chord_window_grid(
    *,
    measure_duration: Fraction,
    total_duration: Fraction,
    resolution: int,
) -> tuple[tuple[Fraction, Fraction], ...]:
    window_value = Fraction(1, resolution)
    windows: list[tuple[Fraction, Fraction]] = []
    window_start = Fraction(0)
    while window_start < total_duration:
        next_bar_boundary = (window_start // measure_duration + 1) * measure_duration
        window_end = min(window_start + window_value, next_bar_boundary, total_duration)
        if window_end <= window_start:
            break

        windows.append((window_start, window_end))
        window_start = window_end

    return tuple(windows)
