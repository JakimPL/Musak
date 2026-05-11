import json
import pathlib
from typing import Any

from config.defaults import (
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
from config.models import IntervalsConfig
from core.intervals.schema import (
    IntervalConfigResponse,
    IntervalRequest,
    IntervalResponse,
)
from core.schemas.common import FieldGroupSchema, FieldSchema
from modules.chords.exporter import to_abjad
from modules.chords.generator import get_random_interval
from modules.chords.interval import Interval
from paths import INTERVALS_CONFIG
from shared.directory import create_directory
from shared.exporter import Exporter
from shared.files import load_yaml


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
                    type="integer",
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
                    type="integer",
                    label="Lowest note",
                    default=defaults.get("lowest_note", LOWEST_NOTE),
                    min=MIN_LOWEST_NOTE,
                    max=MAX_LOWEST_NOTE,
                ),
                FieldSchema(
                    name="highest_note",
                    type="integer",
                    label="Highest note",
                    default=defaults.get("highest_note", HIGHEST_NOTE),
                    min=MIN_HIGHEST_NOTE,
                    max=MAX_HIGHEST_NOTE,
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
    def _write_interval_info(interval: Interval, directory: pathlib.Path) -> None:
        data = interval._asdict()
        data["base_note"] = interval.get_base_note_name()
        data["name"] = interval.name
        with open(directory / "interval.json", "w", encoding="utf-8") as f:
            json.dump(data, f)

    def generate(self, request: IntervalRequest) -> IntervalResponse:
        intervals = self._resolve_intervals(request)

        interval = get_random_interval(
            intervals,
            lowest_note=request.lowest_note,
            highest_note=request.highest_note,
        )

        score = to_abjad(
            interval.chord,
            tempo=request.tempo,
            sequential=request.sequential,
        )

        uuid64, directory = create_directory()
        self._write_interval_info(interval, directory)
        Exporter("interval").export(score, directory)

        return IntervalResponse(
            directory=uuid64,
            audio_source="interval.mp3",
            image_source="interval.png",
            interval_info="interval.json",
            intervals=intervals,
        )
