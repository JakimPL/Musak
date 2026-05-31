from dataclasses import dataclass
from typing import Any, Final

import pandas as pd

from musak_model.harmony.schema import Chord, ChordQuality
from musak_model.synthetic.substitution import GenerationTrace
from musak_model.tokens.schema import Hand
from notebooks.utils.piano_roll import PitchSpelling, midi_pitch_name, pitch_label_expression

_LEFT_HAND_COLOR = "#1f77b4"
_RIGHT_HAND_COLOR = "#ff7f0e"
_CHORD_LABEL_COLOR = "#1a237e"
_CHORD_RULE_COLOR = "#9aa0b4"
_CHORD_BAND_FILLS: Final[tuple[str, str]] = ("#dfe3ee", "#eef1f8")
_ACCENT_WEIGHT_DOMAIN = (0.0, 1.0)
_CHORD_WINDOW_COLUMNS: Final[tuple[str, ...]] = ("start_in_bars", "end_in_bars", "mid_in_bars", "band", "label")
_ROMAN_NUMERALS: Final[tuple[str, ...]] = ("I", "II", "III", "IV", "V", "VI", "VII")
_ACCIDENTAL_PREFIX: Final[dict[int, str]] = {-1: "♭", 0: "", 1: "♯"}
_QUALITY_SUFFIX: Final[dict[ChordQuality, str]] = {ChordQuality.DIMINISHED: "°", ChordQuality.AUGMENTED: "+"}
_MINOR_QUALITIES: Final[frozenset[ChordQuality]] = frozenset({ChordQuality.MINOR, ChordQuality.DIMINISHED})


def chord_label(chord: Chord) -> str:
    numeral = _ROMAN_NUMERALS[(chord.root_degree - 1) % len(_ROMAN_NUMERALS)]
    if chord.quality in _MINOR_QUALITIES:
        numeral = numeral.lower()

    return f"{_ACCIDENTAL_PREFIX[chord.root_accidental]}{numeral}{_QUALITY_SUFFIX.get(chord.quality, '')}"


@dataclass(frozen=True)
class BaselineOverlayViewData:
    pitch_curve: pd.DataFrame
    impulse_grid: pd.DataFrame
    chord_windows: pd.DataFrame
    bar_domain: tuple[float, float]
    pitch_domain: tuple[float, float]
    pitch_spelling: PitchSpelling


def baseline_overlay_view_data(
    trace: GenerationTrace,
    *,
    pitch_spelling: PitchSpelling = PitchSpelling.SHARPS,
) -> BaselineOverlayViewData:
    pitch_rows = []
    impulse_rows = []
    for sample in trace.samples:
        pitch_rows.append(
            {
                "hand": sample.hand.value,
                "bar_index": sample.bar_index,
                "start_in_bars": sample.start_in_bars,
                "register_anchor": sample.register_anchor,
                "register_midi_pitch": sample.register_midi_pitch,
                "pitch": midi_pitch_name(sample.register_midi_pitch, pitch_spelling=pitch_spelling),
            }
        )
        impulse_rows.append(
            {
                "hand": sample.hand.value,
                "bar_index": sample.bar_index,
                "start_in_bars": sample.start_in_bars,
                "accent_weight": sample.accent_weight,
            }
        )

    chord_rows = [
        {
            "start_in_bars": window.start_in_bars,
            "end_in_bars": window.end_in_bars,
            "mid_in_bars": (window.start_in_bars + window.end_in_bars) / 2,
            "band": str(index % 2),
            "label": chord_label(window.chord),
        }
        for index, window in enumerate(trace.chord_windows)
    ]

    pitch_curve = pd.DataFrame(pitch_rows)
    impulse_grid = pd.DataFrame(impulse_rows)
    chord_windows = pd.DataFrame(chord_rows, columns=list(_CHORD_WINDOW_COLUMNS))
    bar_domain = (1.0, float(trace.bar_count + 1))
    if pitch_curve.empty:
        pitch_domain = (0.0, 0.0)
    else:
        pitch_domain = (
            float(pitch_curve["register_midi_pitch"].min()) - 1,
            float(pitch_curve["register_midi_pitch"].max()) + 1,
        )

    return BaselineOverlayViewData(
        pitch_curve=pitch_curve,
        impulse_grid=impulse_grid,
        chord_windows=chord_windows,
        bar_domain=bar_domain,
        pitch_domain=pitch_domain,
        pitch_spelling=pitch_spelling,
    )


def baseline_overlay_chart(
    view_data: BaselineOverlayViewData,
    *,
    alt: Any,
    height: int = 400,
) -> Any:
    color = alt.Color(
        "hand:N",
        title="Hand",
        scale=alt.Scale(
            domain=[Hand.LEFT.value, Hand.RIGHT.value],
            range=[_LEFT_HAND_COLOR, _RIGHT_HAND_COLOR],
        ),
    )
    shared_x = alt.X(
        "start_in_bars:Q",
        title="Bars",
        axis=alt.Axis(grid=True),
        scale=alt.Scale(domain=list(view_data.bar_domain)),
    )
    pitch_base = alt.Chart(view_data.pitch_curve).encode(
        x=shared_x,
        y=alt.Y(
            "register_midi_pitch:Q",
            title="Register baseline (pitch)",
            axis=alt.Axis(labelExpr=pitch_label_expression(view_data.pitch_spelling)),
            scale=alt.Scale(domain=list(view_data.pitch_domain)),
        ),
        color=color,
        tooltip=[
            alt.Tooltip("hand:N", title="Hand"),
            alt.Tooltip("pitch:N", title="Pitch"),
            alt.Tooltip("register_midi_pitch:Q", title="MIDI"),
            alt.Tooltip("register_anchor:Q", title="Anchor"),
            alt.Tooltip("start_in_bars:Q", title="Bar", format=".3f"),
        ],
    )
    pitch_line = pitch_base.mark_line().encode(detail=alt.Detail("hand:N"))
    pitch_points = pitch_base.mark_point(filled=True)
    pitch_curve = pitch_line + pitch_points
    impulse_grid = (
        alt.Chart(view_data.impulse_grid)
        .mark_rule()
        .encode(
            x=shared_x,
            y=alt.Y(
                "accent_weight:Q",
                title="Metric baseline (accent weight)",
                axis=alt.Axis(orient="right"),
                scale=alt.Scale(domain=list(_ACCENT_WEIGHT_DOMAIN)),
            ),
            color=color,
            tooltip=[
                alt.Tooltip("hand:N", title="Hand"),
                alt.Tooltip("accent_weight:Q", title="Accent weight", format=".3f"),
                alt.Tooltip("start_in_bars:Q", title="Bar", format=".3f"),
            ],
        )
    )
    chord_band_x = alt.X("start_in_bars:Q", scale=alt.Scale(domain=list(view_data.bar_domain)))
    chord_tooltip = [
        alt.Tooltip("label:N", title="Chord"),
        alt.Tooltip("start_in_bars:Q", title="From bar", format=".3f"),
        alt.Tooltip("end_in_bars:Q", title="To bar", format=".3f"),
    ]
    chord_bands = (
        alt.Chart(view_data.chord_windows)
        .mark_rect(opacity=0.55)
        .encode(
            x=chord_band_x,
            x2="end_in_bars:Q",
            fill=alt.Fill(
                "band:N",
                scale=alt.Scale(domain=["0", "1"], range=list(_CHORD_BAND_FILLS)),
                legend=None,
            ),
            tooltip=chord_tooltip,
        )
    )
    chord_rules = (
        alt.Chart(view_data.chord_windows)
        .mark_rule(color=_CHORD_RULE_COLOR, strokeDash=[3, 3], opacity=0.8)
        .encode(x=chord_band_x, tooltip=chord_tooltip)
    )
    chord_labels = (
        alt.Chart(view_data.chord_windows)
        .mark_text(baseline="top", dy=4, fontSize=14, fontWeight="bold", color=_CHORD_LABEL_COLOR)
        .encode(
            x=alt.X("mid_in_bars:Q", scale=alt.Scale(domain=list(view_data.bar_domain))),
            y=alt.value(0),
            text=alt.Text("label:N"),
            tooltip=chord_tooltip,
        )
    )
    return (
        alt.layer(chord_bands, chord_rules, pitch_curve, impulse_grid, chord_labels)
        .resolve_scale(y="independent")
        .properties(width="container", height=height, title="Baseline overlay (register, accent, chord track)")
    )
