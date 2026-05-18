from __future__ import annotations

import html
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Final, Literal

from musak.core.notation.schema import (
    EIGHTH,
    HALF,
    QUARTER,
    REST_SUFFIX,
    SIXTEENTH,
    THIRTY_SECOND,
    WHOLE,
    NoteData,
    ScoreData,
    StaveData,
    VexflowDuration,
    VoiceData,
)
from musak.modules.elements.names import midi_to_vexflow_key
from musak_model.data.schema import Segment
from musak_model.decoder import PianoRollEvent, segment_to_piano_roll_events
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand

_POWER_DURATION_BY_FRACTION: Final[dict[Fraction, str]] = {
    Fraction(1, 1): WHOLE,
    Fraction(1, 2): HALF,
    Fraction(1, 4): QUARTER,
    Fraction(1, 8): EIGHTH,
    Fraction(1, 16): SIXTEENTH,
    Fraction(1, 32): THIRTY_SECOND,
}
_MAX_DOTS: Final[int] = 2
_REST_KEY: Final[str] = "b/4"
_NOTE_KIND: Final[Literal["note"]] = "note"
_REST_KIND: Final[Literal["rest"]] = "rest"
_NOTATION_JS_PATH: Final[Path] = Path(__file__).parents[2] / "static" / "js" / "shared" / "notation.js"


class UnsupportedNotationDurationError(ValueError):
    """Raised when a decoded duration cannot be represented by the current VexFlow schema."""


@dataclass(frozen=True)
class ModelNotationEvent:
    kind: Literal["note", "rest"]
    start: Fraction
    duration: Fraction
    midi_pitches: tuple[int, ...] = ()
    tie_start: bool = False
    tie_stop: bool = False

    @property
    def end(self) -> Fraction:
        return self.start + self.duration


def segment_to_score_data(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
    tempo: int | None = None,
    measures_per_row: int | None = None,
    max_bars: int | None = None,
) -> ScoreData:
    events = tuple(segment_to_piano_roll_events(segment, duration_vocabulary=duration_vocabulary))
    measure_duration = Fraction(segment.time_numerator, segment.time_denominator)
    displayed_bar_count = segment.bar_count if max_bars is None else min(segment.bar_count, max_bars)
    rows: list[list[StaveData]] = []
    for first_measure in range(0, displayed_bar_count, measures_per_row or max(displayed_bar_count, 1)):
        last_measure = min(first_measure + (measures_per_row or displayed_bar_count), displayed_bar_count)
        rows.extend(
            _score_rows_for_measure_range(
                events,
                measure_duration=measure_duration,
                first_measure=first_measure,
                last_measure=last_measure,
                time_signature=(segment.time_numerator, segment.time_denominator),
            )
        )

    return ScoreData(rows=rows, tempo=tempo, max_notes_per_measure=_max_notes_per_measure(rows))


def score_data_html(score_data: ScoreData, *, element_id: str = "model-output-notation") -> str:
    score_payload = html.escape(json.dumps(score_data.model_dump(mode="json")), quote=False)
    escaped_id = html.escape(element_id, quote=True)
    module_source = _NOTATION_JS_PATH.read_text(encoding="utf-8")
    return f"""
    <!doctype html>
    <html>
    <head>
      <style>
        body {{
          margin: 0;
          font-family: sans-serif;
        }}
      </style>
    </head>
    <body>
    <div id="{escaped_id}" style="width: 100%; overflow-x: auto;">Loading notation...</div>
    <script type="module">
      const container = document.getElementById({json.dumps(element_id)});
      try {{
        const moduleSource = {json.dumps(module_source)};
        const moduleUrl = URL.createObjectURL(new Blob([moduleSource], {{ type: 'text/javascript' }}));
        const {{ renderScore }} = await import(moduleUrl);
        const scoreData = JSON.parse({json.dumps(score_payload)});
        renderScore(scoreData, container);
        URL.revokeObjectURL(moduleUrl);
      }} catch (error) {{
        container.textContent = `Notation render failed: ${{error.message}}`;
      }}
    </script>
    </body>
    </html>
    """


def _score_rows_for_measure_range(
    events: tuple[PianoRollEvent, ...],
    *,
    measure_duration: Fraction,
    first_measure: int,
    last_measure: int,
    time_signature: tuple[int, int],
) -> list[list[StaveData]]:
    rows: list[list[StaveData]] = []
    right_staves = _staves_for_hand(
        events,
        hand=Hand.RIGHT,
        clef="treble",
        measure_duration=measure_duration,
        first_measure=first_measure,
        last_measure=last_measure,
        time_signature=time_signature,
    )
    left_staves = _staves_for_hand(
        events,
        hand=Hand.LEFT,
        clef="bass",
        measure_duration=measure_duration,
        first_measure=first_measure,
        last_measure=last_measure,
        time_signature=time_signature,
    )
    rows.append(right_staves)
    rows.append(left_staves)

    return rows


def _staves_for_hand(
    events: tuple[PianoRollEvent, ...],
    *,
    hand: Hand,
    clef: str,
    measure_duration: Fraction,
    first_measure: int,
    last_measure: int,
    time_signature: tuple[int, int],
) -> list[StaveData]:
    staves: list[StaveData] = []
    hand_events = tuple(event for event in events if event.hand == hand)
    for measure_index in range(first_measure, last_measure):
        notation_events = _measure_notation_events(
            hand_events,
            measure_index=measure_index,
            measure_duration=measure_duration,
        )
        staves.append(
            StaveData(
                clef=clef,
                time_signature=time_signature if measure_index == first_measure else None,
                voices=[VoiceData(notes=[_notation_event_to_note_data(event) for event in notation_events])],
            )
        )

    return staves


def _measure_notation_events(
    events: tuple[PianoRollEvent, ...],
    *,
    measure_index: int,
    measure_duration: Fraction,
) -> list[ModelNotationEvent]:
    measure_start = measure_index * measure_duration
    measure_end = measure_start + measure_duration
    fragments: list[ModelNotationEvent] = []
    for start, event_group in _group_events_by_onset(events):
        group_end = max(event.end for event in event_group)
        if group_end <= measure_start or start >= measure_end:
            continue

        fragment_start = max(start, measure_start)
        fragment_end = min(group_end, measure_end)
        if fragment_start >= fragment_end:
            continue

        fragments.append(
            ModelNotationEvent(
                kind=_NOTE_KIND,
                start=fragment_start - measure_start,
                duration=fragment_end - fragment_start,
                midi_pitches=tuple(sorted(event.midi_pitch for event in event_group)),
                tie_start=fragment_end < group_end,
                tie_stop=fragment_start > start,
            )
        )

    return _fill_rests(sorted(fragments, key=lambda event: (event.start, event.midi_pitches)), measure_duration)


def _group_events_by_onset(events: tuple[PianoRollEvent, ...]) -> list[tuple[Fraction, tuple[PianoRollEvent, ...]]]:
    groups: dict[Fraction, list[PianoRollEvent]] = {}
    for event in events:
        groups.setdefault(event.start, []).append(event)

    return [(start, tuple(group)) for start, group in sorted(groups.items())]


def _fill_rests(events: list[ModelNotationEvent], measure_duration: Fraction) -> list[ModelNotationEvent]:
    output: list[ModelNotationEvent] = []
    cursor = Fraction(0)
    for event in events:
        if event.start > cursor:
            output.append(ModelNotationEvent(kind=_REST_KIND, start=cursor, duration=event.start - cursor))
        output.append(event)
        cursor = max(cursor, event.end)

    if cursor < measure_duration:
        output.append(ModelNotationEvent(kind=_REST_KIND, start=cursor, duration=measure_duration - cursor))

    return output


def _notation_event_to_note_data(event: ModelNotationEvent) -> NoteData:
    duration, dots = _duration_to_vexflow(event.duration)
    if event.kind == _REST_KIND:
        return NoteData(keys=[_REST_KEY], duration=_rest_duration(duration), dots=dots)

    return NoteData(
        keys=[midi_to_vexflow_key(midi_pitch) for midi_pitch in event.midi_pitches],
        duration=duration,
        dots=dots,
        tie_start=event.tie_start,
        tie_stop=event.tie_stop,
    )


def _duration_to_vexflow(duration: Fraction) -> tuple[VexflowDuration, int]:
    for dots in range(_MAX_DOTS + 1):
        multiplier = Fraction(2 ** (dots + 1) - 1, 2**dots)
        base_duration = duration / multiplier
        if base_duration in _POWER_DURATION_BY_FRACTION:
            return _POWER_DURATION_BY_FRACTION[base_duration], dots  # type: ignore[return-value]

    raise UnsupportedNotationDurationError(f"duration {duration} is not supported by the VexFlow notebook bridge")


def _rest_duration(duration: VexflowDuration) -> VexflowDuration:
    return f"{duration}{REST_SUFFIX}"  # type: ignore[return-value]


def _max_notes_per_measure(rows: list[list[StaveData]]) -> int | None:
    counts = [len(stave.voices[0].notes) for row in rows for stave in row if stave.voices]
    if not counts:
        return None

    return max(counts)
