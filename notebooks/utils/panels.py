import hashlib
import html
from dataclasses import dataclass

from musak_model.tokens.schema import Hand
from musak_shared.exporter import AudioExportError, Exporter
from notebooks.utils.audio import piano_roll_events_to_audio_data
from notebooks.utils.piano_roll import PianoRollViewData, filter_piano_roll_dataframe, piano_roll_chart


@dataclass(frozen=True)
class HandControls:
    right: object
    left: object

    @property
    def selected_hands(self) -> frozenset[Hand]:
        selected = set()
        if self.right.value:
            selected.add(Hand.RIGHT)
        if self.left.value:
            selected.add(Hand.LEFT)
        return frozenset(selected)


def hand_controls(mo: object, *, right: bool = True, left: bool = True) -> HandControls:
    return HandControls(
        right=mo.ui.checkbox(value=right, label="Right hand"),
        left=mo.ui.checkbox(value=left, label="Left hand"),
    )


def piano_roll_player_panel(
    view_data: PianoRollViewData | None,
    *,
    mo: object,
    alt: object,
    bpm: int,
    controls: HandControls,
    exporter: Exporter | None = None,
) -> object:
    if view_data is None:
        return mo.md("")

    selected_hands = controls.selected_hands
    controls_output = mo.hstack([controls.right, controls.left], justify="start", gap=2)
    if not selected_hands:
        return mo.vstack(
            [
                controls_output,
                mo.callout("Select at least one hand to display and play notes.", kind="warn"),
            ],
            gap=1,
        )

    frame = filter_piano_roll_dataframe(view_data.dataframe, hands=selected_hands)
    if frame.empty:
        return mo.vstack(
            [
                controls_output,
                mo.callout("No note events decoded for the selected hand(s).", kind="warn"),
            ],
            gap=1,
        )

    try:
        audio_data = piano_roll_events_to_audio_data(
            view_data.events,
            bpm=bpm,
            hands=selected_hands,
            exporter=exporter,
        )
        player_output = audio_player(mo, audio_data, selected_hands=selected_hands)
    except AudioExportError as exception:
        player_output = mo.callout(f"Audio export failed: {exception}", kind="warn")

    chart_output = mo.ui.altair_chart(
        piano_roll_chart(view_data, alt=alt, hands=selected_hands),
        chart_selection=False,
        legend_selection=False,
    )
    return mo.vstack([controls_output, player_output, chart_output], gap=1)


def audio_player(mo: object, audio_data: str, *, selected_hands: frozenset[Hand]) -> object:
    hand_key = "-".join(hand.value for hand in Hand if hand in selected_hands)
    audio_hash = hashlib.sha256(audio_data.encode("utf-8")).hexdigest()[:16]
    element_id = f"piano-roll-audio-{hand_key}-{audio_hash}"
    escaped_audio_data = html.escape(audio_data, quote=True)
    return mo.Html(f"""
        <div id="{element_id}" data-audio-key="{element_id}">
            <audio controls preload="metadata" src="{escaped_audio_data}"></audio>
        </div>
        """)
