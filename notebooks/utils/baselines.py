from dataclasses import dataclass
from typing import Any

import pandas as pd

from musak_model.synthetic.substitution import GenerationTrace
from musak_model.tokens.schema import Hand
from notebooks.utils.piano_roll import PitchSpelling, midi_pitch_name, pitch_label_expression

_LEFT_HAND_COLOR = "#1f77b4"
_RIGHT_HAND_COLOR = "#ff7f0e"
_ACCENT_WEIGHT_DOMAIN = (0.0, 1.0)


@dataclass(frozen=True)
class BaselineOverlayViewData:
    pitch_curve: pd.DataFrame
    impulse_grid: pd.DataFrame
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

    pitch_curve = pd.DataFrame(pitch_rows)
    impulse_grid = pd.DataFrame(impulse_rows)
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
    return (
        alt.layer(pitch_curve, impulse_grid)
        .resolve_scale(y="independent")
        .properties(width="container", height=height, title="Baseline overlay")
    )
