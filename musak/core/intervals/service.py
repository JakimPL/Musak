from typing import Any

from musak.config.defaults import (
    HIGHEST_NOTE,
    LOWEST_NOTE,
    MAX_HIGHEST_NOTE,
    MAX_LOWEST_NOTE,
    MAX_TEMPO,
    MIN_HIGHEST_NOTE,
    MIN_LOWEST_NOTE,
    MIN_TEMPO,
    SEQUENTIAL,
    TEMPO,
)
from musak.config.models import IntervalsConfig
from musak.core.intervals.schema import (
    IntervalConfigResponse,
    IntervalRequest,
    IntervalResponse,
)
from musak.core.notation.chord_serializer import interval_to_score_data
from musak.core.schemas.common import FieldGroupSchema, FieldSchema
from musak.modules.chords.exporter import to_midi
from musak.modules.chords.generator import get_random_interval
from musak.modules.elements.interval import Interval
from musak.paths import INTERVALS_CONFIG
from musak.shared.dict import namedtuple_with_base_note
from musak.shared.exporter import midi_to_audio
from musak.shared.files import load_yaml


class IntervalService:
    def __init__(self) -> None:
        self._definitions = self._load_definitions()

    def _load_config(self) -> IntervalsConfig:
        return load_yaml(INTERVALS_CONFIG, IntervalsConfig)

    def _load_definitions(self) -> dict[str, int]:
        return self._load_config().intervals_definitions

    def _load_defaults(self) -> dict[str, Any]:
        return self._load_config().default_settings.model_dump()

    @property
    def definitions(self) -> dict[str, int]:
        return self._definitions

    def get_config(self) -> IntervalConfigResponse:
        defaults = self._load_defaults()

        options_group = FieldGroupSchema(
            label="Options",
            fields=[
                FieldSchema(
                    name="sequential",
                    type="boolean",
                    label="Sequential",
                    default=defaults.get("sequential", SEQUENTIAL),
                ),
            ],
        )

        tempo_group = FieldGroupSchema(
            label="Tempo",
            fields=[
                FieldSchema(
                    name="tempo",
                    type="slider",
                    label="Tempo",
                    default=defaults.get("tempo", TEMPO),
                    min=MIN_TEMPO,
                    max=MAX_TEMPO,
                ),
            ],
        )

        range_group = FieldGroupSchema(
            label="Notes range",
            fields=[
                FieldSchema(
                    name="lowest_note",
                    type="slider",
                    label="Lowest note",
                    default=defaults.get("lowest_note", LOWEST_NOTE),
                    min=MIN_LOWEST_NOTE,
                    max=MAX_LOWEST_NOTE,
                    format="note",
                ),
                FieldSchema(
                    name="highest_note",
                    type="slider",
                    label="Highest note",
                    default=defaults.get("highest_note", HIGHEST_NOTE),
                    min=MIN_HIGHEST_NOTE,
                    max=MAX_HIGHEST_NOTE,
                    format="note",
                ),
            ],
        )

        interval_fields = [
            FieldSchema(
                name=f"interval_{name}",
                type="boolean",
                label=name.replace("_", " "),
                default=(defaults.get(f"interval_{name}") == "on"),
            )
            for name in self._definitions
        ]
        intervals_group = FieldGroupSchema(label="Intervals", fields=interval_fields)

        return IntervalConfigResponse(
            groups=[options_group, tempo_group, range_group, intervals_group],
            definitions=self._definitions,
        )

    def _resolve_intervals(self, request: IntervalRequest) -> dict[str, int]:
        return request.intervals if request.intervals else self._definitions

    @staticmethod
    def _build_interval_info(interval: Interval) -> dict[str, Any]:
        data = namedtuple_with_base_note(interval)
        data["name"] = interval.name
        return data

    def generate(self, request: IntervalRequest) -> IntervalResponse:
        intervals = self._resolve_intervals(request)

        interval = get_random_interval(
            intervals,
            lowest_note=request.lowest_note,
            highest_note=request.highest_note,
        )

        score_data = interval_to_score_data(interval, sequential=request.sequential, tempo=request.tempo)

        midi_file = to_midi(
            interval.chord,
            tempo=request.tempo,
            sequential=request.sequential,
        )
        audio_data = midi_to_audio(midi_file)

        return IntervalResponse(
            audio_data=audio_data,
            score_data=score_data,
            interval_info=self._build_interval_info(interval),
            intervals=intervals,
        )
