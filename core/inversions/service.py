import json
import pathlib
from typing import Any, Mapping

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
from config.models import InversionsConfig
from core.inversions.schema import (
    InversionConfigResponse,
    InversionRequest,
    InversionResponse,
)
from core.schemas.common import FieldGroupSchema, FieldSchema
from modules.chords.exporter import to_abjad
from modules.chords.generator import generate_all_inversions, get_random_chord_inversion
from paths import INVERSIONS_CONFIG
from shared.directory import create_directory
from shared.exporter import Exporter
from shared.files import load_yaml

CHORD_NAMES = {
    "": "major",
    "m": "minor",
    "dim": "diminished",
    "aug": "augmented",
    "sus4": "suspended (4)",
    "7": "dominant",
    "maj7": "major seventh",
    "m7": "minor seventh",
    "m(maj7)": "minor major seventh",
}


class InversionService:
    def __init__(self) -> None:
        self._definitions = self._load_definitions()

    def _load_config(self) -> InversionsConfig:
        return load_yaml(INVERSIONS_CONFIG, InversionsConfig)

    def _load_definitions(self) -> dict[str, list[int]]:
        return self._load_config().chords_definitions

    def _load_defaults(self) -> dict[str, Any]:
        return self._load_config().default_settings.model_dump()

    @property
    def definitions(self) -> dict[str, list[int]]:
        return self._definitions

    def get_config(self) -> InversionConfigResponse:
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

        chord_fields = [
            FieldSchema(
                name=f"chord_{name}",
                type="boolean",
                label=CHORD_NAMES.get(name, name),
                default=(defaults.get(f"chord_{name}") == "on"),
            )
            for name in self._definitions
        ]
        chords_group = FieldGroupSchema(label="Chords", fields=chord_fields)

        return InversionConfigResponse(
            groups=[options_group, tempo_group, range_group, chords_group],
            definitions=self._definitions,
        )

    def _resolve_chords(self, request: InversionRequest) -> Mapping[str, list[int]]:
        return request.chords if request.chords else self._definitions

    @staticmethod
    def _write_chord_info(chord_inversion: Any, directory: pathlib.Path) -> None:
        data = chord_inversion._asdict()
        data["base_note"] = chord_inversion.get_base_note_name()
        with open(directory / "chord.json", "w", encoding="utf-8") as f:
            json.dump(data, f)

    def generate(self, request: InversionRequest) -> InversionResponse:
        chords = self._resolve_chords(request)

        inversions = generate_all_inversions(dict(chords))
        chord_inversion = get_random_chord_inversion(
            inversions,
            lowest_note=request.lowest_note,
            highest_note=request.highest_note,
        )
        score = to_abjad(
            chord_inversion.chord,
            tempo=request.tempo,
            sequential=request.sequential,
        )

        uuid64, directory = create_directory()
        self._write_chord_info(chord_inversion, directory)
        Exporter("chord").export(score, directory)

        inversions_numbers = {chord_type: len(inv_list) for chord_type, inv_list in inversions.items()}

        return InversionResponse(
            directory=uuid64,
            audio_source="chord.mp3",
            image_source="chord.png",
            chord_info="chord.json",
            chord_types=list(chords.keys()),
            inversions_numbers=inversions_numbers,
        )
