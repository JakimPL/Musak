from musak_model.synthetic.substitution import BaselineSample, GenerationTrace
from musak_model.tokens.schema import Hand
from notebooks.utils.baselines import baseline_overlay_view_data


def _trace() -> GenerationTrace:
    samples = (
        BaselineSample(
            hand=Hand.RIGHT,
            bar_index=0,
            position=0,
            start_in_bars=1.0,
            register_anchor=0,
            register_midi_pitch=72,
            accent_weight=0.5,
        ),
        BaselineSample(
            hand=Hand.LEFT,
            bar_index=0,
            position=0,
            start_in_bars=1.0,
            register_anchor=0,
            register_midi_pitch=48,
            accent_weight=0.25,
        ),
        BaselineSample(
            hand=Hand.RIGHT,
            bar_index=1,
            position=0,
            start_in_bars=2.0,
            register_anchor=2,
            register_midi_pitch=76,
            accent_weight=0.75,
        ),
        BaselineSample(
            hand=Hand.LEFT,
            bar_index=1,
            position=0,
            start_in_bars=2.0,
            register_anchor=1,
            register_midi_pitch=50,
            accent_weight=0.1,
        ),
    )
    return GenerationTrace(samples=samples, grid_count_per_bar=1, bar_count=2)


def test_view_data_columns_and_shape() -> None:
    view_data = baseline_overlay_view_data(_trace())

    assert len(view_data.pitch_curve) == 4
    assert len(view_data.impulse_grid) == 4
    assert set(view_data.pitch_curve.columns) == {
        "hand",
        "bar_index",
        "start_in_bars",
        "register_anchor",
        "register_midi_pitch",
        "pitch",
    }
    assert set(view_data.impulse_grid.columns) == {"hand", "bar_index", "start_in_bars", "accent_weight"}


def test_view_data_domains() -> None:
    view_data = baseline_overlay_view_data(_trace())

    assert view_data.bar_domain == (1.0, 3.0)
    assert view_data.pitch_domain == (47.0, 77.0)


def test_view_data_handles_empty_trace() -> None:
    view_data = baseline_overlay_view_data(GenerationTrace(samples=(), grid_count_per_bar=1, bar_count=0))

    assert view_data.pitch_curve.empty
    assert view_data.impulse_grid.empty
    assert view_data.bar_domain == (1.0, 1.0)
