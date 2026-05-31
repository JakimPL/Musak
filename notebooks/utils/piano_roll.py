from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from math import ceil, floor
from typing import Any, Final

import pandas as pd

from musak_model.data.schema import ParsedScore, Segment
from musak_model.decoder import PianoRollEvent, parsed_score_to_piano_roll_events, segment_to_piano_roll_events
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.pitch import degree_pitch_class
from musak_model.tokens.schema import Hand, ScaleType, scale_size_for_type
from musak_shared.elements import MIDI_MAX_PITCH, PITCHES_PER_OCTAVE
from musak_shared.ratios import format_ratio

_QUARTERS_PER_WHOLE: Final[int] = 4
_SECONDS_PER_MINUTE: Final[int] = 60
_LEFT_HAND_COLOR: Final[str] = "#1f77b4"
_RIGHT_HAND_COLOR: Final[str] = "#ff7f0e"
_NOTE_STROKE_COLOR: Final[str] = "#ffffff"
_NOTE_STROKE_WIDTH: Final[float] = 0.7
_NOTE_BAR_MARGIN: Final[Fraction] = Fraction(1, 250)
_SCALE_HIGHLIGHT_FILL: Final[str] = "#43a047"
_SCALE_HIGHLIGHT_STROKE: Final[str] = "#2e7d32"
_SCALE_HIGHLIGHT_FILL_OPACITY: Final[float] = 0.10
_SCALE_HIGHLIGHT_STROKE_OPACITY: Final[float] = 0.35
_CHORD_HIGHLIGHT_FILL: Final[str] = "#7e57c2"
_CHORD_HIGHLIGHT_STROKE: Final[str] = "#4527a0"
_CHORD_HIGHLIGHT_FILL_OPACITY: Final[float] = 0.22
_CHORD_HIGHLIGHT_STROKE_OPACITY: Final[float] = 0.5
_HIGHLIGHT_STROKE_WIDTH: Final[float] = 0.5
_PITCH_BAND_HALF_HEIGHT: Final[float] = 0.5
_SHARP_PITCH_NAMES: Final[tuple[str, ...]] = ("C-", "C#", "D-", "D#", "E-", "F-", "F#", "G-", "G#", "A-", "A#", "B-")
_FLAT_PITCH_NAMES: Final[tuple[str, ...]] = ("C-", "Db", "D-", "Eb", "E-", "F-", "Gb", "G-", "Ab", "A-", "Bb", "B-")


class PitchSpelling(StrEnum):
    SHARPS = "sharps"
    FLATS = "flats"


@dataclass(frozen=True)
class ChordHighlight:
    start_in_bars: float
    end_in_bars: float
    pitch_classes: frozenset[int]
    label: str


def scale_pitch_class_set(scale_root: int, scale_type: ScaleType) -> frozenset[int]:
    return frozenset(
        (scale_root + degree_pitch_class(degree, 0, scale_type=scale_type)) % PITCHES_PER_OCTAVE
        for degree in range(1, scale_size_for_type(scale_type) + 1)
    )


@dataclass(frozen=True)
class PianoRollViewData:
    events: tuple[PianoRollEvent, ...]
    dataframe: pd.DataFrame
    title: str
    bar_domain: tuple[float, float]
    seconds_domain: tuple[float, float]
    pitch_spelling: PitchSpelling


def piano_roll_dataframe(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
    pitch_spelling: PitchSpelling = PitchSpelling.SHARPS,
    bpm: int = 60,
) -> pd.DataFrame:
    events = segment_to_piano_roll_events(segment, duration_vocabulary=duration_vocabulary)
    return _events_to_dataframe(
        events,
        measure_duration=Fraction(segment.time_numerator, segment.time_denominator),
        window_start_bar=segment.metadata.window_start_bar,
        pitch_spelling=pitch_spelling,
        bpm=bpm,
    )


def parsed_score_piano_roll_dataframe(
    score: ParsedScore,
    *,
    pitch_spelling: PitchSpelling = PitchSpelling.SHARPS,
    bpm: int = 60,
) -> pd.DataFrame:
    events = parsed_score_to_piano_roll_events(score)
    return _events_to_dataframe(
        events,
        measure_duration=Fraction(score.time_numerator, score.time_denominator),
        window_start_bar=0,
        pitch_spelling=pitch_spelling,
        bpm=bpm,
    )


def segment_piano_roll_view_data(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
    pitch_spelling: PitchSpelling = PitchSpelling.SHARPS,
    bpm: int = 60,
    title: str = "Decoded segment piano roll",
) -> PianoRollViewData:
    events = tuple(segment_to_piano_roll_events(segment, duration_vocabulary=duration_vocabulary))
    measure_duration = Fraction(segment.time_numerator, segment.time_denominator)
    dataframe = _events_to_dataframe(
        events,
        measure_duration=measure_duration,
        window_start_bar=segment.metadata.window_start_bar,
        pitch_spelling=pitch_spelling,
        bpm=bpm,
    )
    return PianoRollViewData(
        events=events,
        dataframe=dataframe,
        title=title,
        bar_domain=(
            float(segment.metadata.window_start_bar + 1),
            float(segment.metadata.window_start_bar + segment.bar_count + 1),
        ),
        seconds_domain=(0.0, _whole_note_fraction_to_seconds(measure_duration * segment.bar_count, bpm=bpm)),
        pitch_spelling=pitch_spelling,
    )


def parsed_score_piano_roll_view_data(
    score: ParsedScore,
    *,
    pitch_spelling: PitchSpelling = PitchSpelling.SHARPS,
    bpm: int = 60,
    title: str = "Parsed score piano roll",
) -> PianoRollViewData:
    events = tuple(parsed_score_to_piano_roll_events(score))
    measure_duration = Fraction(score.time_numerator, score.time_denominator)
    bar_count = max(len(score.right_hand_bars), len(score.left_hand_bars))
    dataframe = _events_to_dataframe(
        events,
        measure_duration=measure_duration,
        window_start_bar=0,
        pitch_spelling=pitch_spelling,
        bpm=bpm,
    )
    return PianoRollViewData(
        events=events,
        dataframe=dataframe,
        title=title,
        bar_domain=(1.0, float(bar_count + 1)),
        seconds_domain=(0.0, _whole_note_fraction_to_seconds(measure_duration * bar_count, bpm=bpm)),
        pitch_spelling=pitch_spelling,
    )


def filter_piano_roll_dataframe(frame: pd.DataFrame, *, hands: frozenset[Hand]) -> pd.DataFrame:
    if frame.empty:
        return frame

    hand_values = {hand.value for hand in hands}
    return frame[frame["hand"].isin(hand_values)].copy()


def piano_roll_chart(
    view_data: PianoRollViewData,
    *,
    alt: Any,
    hands: frozenset[Hand] = frozenset(Hand),
    height: int = 400,
    scale_pitch_classes: frozenset[int] | None = None,
    chord_highlights: Sequence[ChordHighlight] = (),
) -> Any:
    frame = filter_piano_roll_dataframe(view_data.dataframe, hands=hands)
    label_expression = pitch_label_expression(view_data.pitch_spelling)
    y_domain = [
        max(0, float(frame["midi_pitch"].min()) - 1),
        min(MIDI_MAX_PITCH, float(frame["midi_pitch"].max()) + 1),
    ]
    note_bars = (
        alt.Chart(frame)
        .mark_bar(stroke=_NOTE_STROKE_COLOR, strokeWidth=_NOTE_STROKE_WIDTH)
        .encode(
            x=alt.X(
                "bar_start_display:Q",
                title="Bars",
                axis=alt.Axis(grid=True),
                scale=alt.Scale(domain=list(view_data.bar_domain)),
            ),
            x2="bar_end_display:Q",
            y=alt.Y(
                "midi_pitch:Q",
                title="Pitch",
                axis=alt.Axis(labelExpr=label_expression),
                scale=alt.Scale(domain=y_domain),
            ),
            color=alt.Color(
                "hand:N",
                title="Hand",
                scale=alt.Scale(
                    domain=[Hand.LEFT.value, Hand.RIGHT.value],
                    range=[_LEFT_HAND_COLOR, _RIGHT_HAND_COLOR],
                ),
            ),
            tooltip=[
                alt.Tooltip("hand:N", title="Hand"),
                alt.Tooltip("pitch:N", title="Pitch"),
                alt.Tooltip("midi_pitch:Q", title="MIDI"),
                alt.Tooltip("bar_start:Q", title="Bar start", format=".3f"),
                alt.Tooltip("bar_end:Q", title="Bar end", format=".3f"),
                alt.Tooltip("start_seconds:Q", title="Start (s)", format=".3f"),
                alt.Tooltip("duration_fraction:N", title="Duration"),
                alt.Tooltip("duration_seconds:Q", title="Duration (s)", format=".3f"),
                alt.Tooltip("token:N", title="Token"),
                alt.Tooltip("token_index:Q", title="Token index"),
            ],
        )
    )
    seconds_axis = (
        alt.Chart(frame)
        .mark_rule(opacity=0)
        .encode(
            x=alt.X(
                "start_seconds:Q",
                title="Time (s)",
                axis=alt.Axis(orient="top", grid=False),
                scale=alt.Scale(domain=list(view_data.seconds_domain)),
            )
        )
    )
    highlight_layers = _highlight_layers(
        alt=alt,
        y_domain=y_domain,
        bar_domain=view_data.bar_domain,
        pitch_spelling=view_data.pitch_spelling,
        scale_pitch_classes=scale_pitch_classes,
        chord_highlights=chord_highlights,
    )
    return (
        alt.layer(*highlight_layers, note_bars, seconds_axis)
        .resolve_scale(x="independent")
        .properties(width="container", height=height, title=view_data.title)
    )


def _highlight_layers(
    *,
    alt: Any,
    y_domain: list[float],
    bar_domain: tuple[float, float],
    pitch_spelling: PitchSpelling,
    scale_pitch_classes: frozenset[int] | None,
    chord_highlights: Sequence[ChordHighlight],
) -> list[Any]:
    layers: list[Any] = []
    if scale_pitch_classes:
        scale_rows = _pitch_band_rows(scale_pitch_classes, y_domain=y_domain, pitch_spelling=pitch_spelling)
        if scale_rows:
            layers.append(
                alt.Chart(pd.DataFrame(scale_rows))
                .mark_rect(
                    fill=_SCALE_HIGHLIGHT_FILL,
                    fillOpacity=_SCALE_HIGHLIGHT_FILL_OPACITY,
                    stroke=_SCALE_HIGHLIGHT_STROKE,
                    strokeWidth=_HIGHLIGHT_STROKE_WIDTH,
                    strokeOpacity=_SCALE_HIGHLIGHT_STROKE_OPACITY,
                )
                .encode(
                    y=alt.Y("pitch_low:Q", axis=None, scale=alt.Scale(domain=y_domain)),
                    y2="pitch_high:Q",
                    tooltip=[alt.Tooltip("pitch:N", title="Scale pitch")],
                )
            )

    chord_rows = [
        {
            **row,
            "start_in_bars": highlight.start_in_bars,
            "end_in_bars": highlight.end_in_bars,
            "label": highlight.label,
        }
        for highlight in chord_highlights
        for row in _pitch_band_rows(highlight.pitch_classes, y_domain=y_domain, pitch_spelling=pitch_spelling)
    ]
    if chord_rows:
        layers.append(
            alt.Chart(pd.DataFrame(chord_rows))
            .mark_rect(
                fill=_CHORD_HIGHLIGHT_FILL,
                fillOpacity=_CHORD_HIGHLIGHT_FILL_OPACITY,
                stroke=_CHORD_HIGHLIGHT_STROKE,
                strokeWidth=_HIGHLIGHT_STROKE_WIDTH,
                strokeOpacity=_CHORD_HIGHLIGHT_STROKE_OPACITY,
            )
            .encode(
                x=alt.X("start_in_bars:Q", axis=None, scale=alt.Scale(domain=list(bar_domain))),
                x2="end_in_bars:Q",
                y=alt.Y("pitch_low:Q", axis=None, scale=alt.Scale(domain=y_domain)),
                y2="pitch_high:Q",
                tooltip=[
                    alt.Tooltip("label:N", title="Chord"),
                    alt.Tooltip("pitch:N", title="Chord pitch"),
                ],
            )
        )

    return layers


def _pitch_band_rows(
    pitch_classes: frozenset[int],
    *,
    y_domain: list[float],
    pitch_spelling: PitchSpelling,
) -> list[dict[str, Any]]:
    return [
        {
            "pitch_low": midi_pitch - _PITCH_BAND_HALF_HEIGHT,
            "pitch_high": midi_pitch + _PITCH_BAND_HALF_HEIGHT,
            "pitch": midi_pitch_name(midi_pitch, pitch_spelling=pitch_spelling),
        }
        for midi_pitch in range(int(floor(y_domain[0])), int(ceil(y_domain[1])) + 1)
        if midi_pitch % PITCHES_PER_OCTAVE in pitch_classes
    ]


def pitch_label_expression(pitch_spelling: PitchSpelling = PitchSpelling.SHARPS) -> str:
    pitch_names = _FLAT_PITCH_NAMES if pitch_spelling == PitchSpelling.FLATS else _SHARP_PITCH_NAMES
    return f"{list(pitch_names)}[datum.value % 12] + floor(datum.value / 12 - 1)"


def midi_pitch_name(midi_pitch: int, *, pitch_spelling: PitchSpelling = PitchSpelling.SHARPS) -> str:
    pitch_names = _FLAT_PITCH_NAMES if pitch_spelling == PitchSpelling.FLATS else _SHARP_PITCH_NAMES
    pitch_class = midi_pitch % len(pitch_names)
    octave = midi_pitch // len(pitch_names) - 1
    return f"{pitch_names[pitch_class]}{octave}"


def _events_to_dataframe(
    events: Iterable[PianoRollEvent],
    *,
    measure_duration: Fraction,
    window_start_bar: int,
    pitch_spelling: PitchSpelling,
    bpm: int,
) -> pd.DataFrame:
    rows = []
    for event in events:
        start_seconds = _whole_note_fraction_to_seconds(event.start, bpm=bpm)
        duration_seconds = _whole_note_fraction_to_seconds(event.duration, bpm=bpm)
        bar_start = window_start_bar + 1 + event.start / measure_duration
        bar_duration = event.duration / measure_duration
        bar_end = bar_start + bar_duration
        bar_margin = min(_NOTE_BAR_MARGIN, bar_duration / 4)
        duration_text = format_ratio(event.duration, separator=":")
        rows.append(
            {
                "hand": event.hand.value,
                "midi_pitch": event.midi_pitch,
                "pitch": midi_pitch_name(event.midi_pitch, pitch_spelling=pitch_spelling),
                "start": float(event.start),
                "duration": float(event.duration),
                "duration_fraction": duration_text,
                "end": float(event.end),
                "bar_start": float(bar_start),
                "bar_end": float(bar_end),
                "bar_start_display": float(bar_start + bar_margin),
                "bar_end_display": float(bar_end - bar_margin),
                "start_seconds": start_seconds,
                "duration_seconds": duration_seconds,
                "end_seconds": start_seconds + duration_seconds,
                "token_index": event.token_index,
                "token": event.token_text,
            }
        )

    return pd.DataFrame(rows)


def _whole_note_fraction_to_seconds(duration: Fraction, *, bpm: int) -> float:
    return float(duration * _QUARTERS_PER_WHOLE * _SECONDS_PER_MINUTE / bpm)
