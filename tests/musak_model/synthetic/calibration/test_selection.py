from musak_model.synthetic.calibration.schema import SweepResult
from musak_model.synthetic.calibration.selection import select_tilts

_GRID = (0.0, 0.5, 1.0)


def _result(lambda_curve: float, lambda_harmonic: float, lambda_accent: float, tv: float | None) -> SweepResult:
    return SweepResult(
        lambda_curve=lambda_curve,
        lambda_harmonic=lambda_harmonic,
        lambda_accent=lambda_accent,
        distribution_groups=1,
        mean_total_variation_distance=tv,
    )


def test_select_tilts_picks_largest_under_threshold_per_direction() -> None:
    results = [
        _result(0.0, 0.0, 0.0, 0.30),
        _result(0.5, 0.0, 0.0, 0.08),
        _result(1.0, 0.0, 0.0, 0.05),
        _result(0.0, 0.5, 0.0, 0.20),
        _result(0.0, 1.0, 0.0, 0.04),
        _result(0.0, 0.0, 0.5, 0.12),
        _result(0.0, 0.0, 1.0, 0.15),
    ]

    selection = select_tilts(results, lambda_curve=_GRID, lambda_harmonic=_GRID, lambda_accent=_GRID, threshold=0.1)

    assert selection.lambda_curve == 1.0
    assert selection.lambda_harmonic == 1.0
    assert selection.lambda_accent == 0.5
    assert selection.threshold_met is False


def test_select_tilts_reports_threshold_met_when_all_directions_clear() -> None:
    results = [
        _result(0.0, 0.0, 0.0, 0.30),
        _result(0.5, 0.0, 0.0, 0.08),
        _result(0.0, 0.5, 0.0, 0.07),
        _result(0.0, 0.0, 0.5, 0.06),
    ]

    selection = select_tilts(results, lambda_curve=_GRID, lambda_harmonic=_GRID, lambda_accent=_GRID, threshold=0.1)

    assert (selection.lambda_curve, selection.lambda_harmonic, selection.lambda_accent) == (0.5, 0.5, 0.5)
    assert selection.threshold_met is True


def test_select_tilts_skips_missing_distances() -> None:
    results = [
        _result(0.0, 0.0, 0.0, None),
        _result(0.5, 0.0, 0.0, 0.05),
        _result(0.0, 0.5, 0.0, None),
        _result(0.0, 0.0, 0.5, None),
    ]

    selection = select_tilts(
        results, lambda_curve=(0.0, 0.5), lambda_harmonic=(0.0, 0.5), lambda_accent=(0.0, 0.5), threshold=0.1
    )

    assert selection.lambda_curve == 0.5
    assert selection.lambda_harmonic == 0.0
    assert selection.lambda_accent == 0.0
    assert selection.threshold_met is False
