from musak_model.harmony.schema import Chord, ChordQuality
from musak_model.synthetic.substitution import BaselineSample, ChordWindowSample, GenerationTrace
from musak_model.tokens.schema import Hand
from notebooks.utils.baselines import baseline_overlay_view_data, chord_label


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


def test_view_data_chord_windows_columns_when_absent() -> None:
    view_data = baseline_overlay_view_data(_trace())

    assert view_data.chord_windows.empty
    assert list(view_data.chord_windows.columns) == ["start_in_bars", "end_in_bars", "mid_in_bars", "band", "label"]


def test_view_data_chord_windows_labels_and_midpoints() -> None:
    windows = (
        ChordWindowSample(
            start_in_bars=1.0,
            end_in_bars=2.0,
            chord=Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR),
        ),
        ChordWindowSample(
            start_in_bars=2.0,
            end_in_bars=3.0,
            chord=Chord(root_degree=5, root_accidental=0, quality=ChordQuality.MAJOR),
        ),
    )
    trace = GenerationTrace(samples=(), grid_count_per_bar=1, bar_count=2, chord_windows=windows)

    chord_windows = baseline_overlay_view_data(trace).chord_windows

    assert list(chord_windows["label"]) == ["I", "V"]
    assert list(chord_windows["mid_in_bars"]) == [1.5, 2.5]


def test_chord_label_quality_and_accidental_spelling() -> None:
    assert chord_label(Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR)) == "I"
    assert chord_label(Chord(root_degree=2, root_accidental=0, quality=ChordQuality.MINOR)) == "ii"
    assert chord_label(Chord(root_degree=7, root_accidental=0, quality=ChordQuality.DIMINISHED)) == "vii°"
    assert chord_label(Chord(root_degree=4, root_accidental=0, quality=ChordQuality.AUGMENTED)) == "IV+"
    assert chord_label(Chord(root_degree=6, root_accidental=-1, quality=ChordQuality.MAJOR)) == "♭VI"
    assert chord_label(Chord(root_degree=4, root_accidental=1, quality=ChordQuality.MINOR)) == "♯iv"
