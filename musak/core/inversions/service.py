from typing import Any, Mapping

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
from musak.config.models import InversionsConfig
from musak.core.inversions.schema import (
    InversionConfigResponse,
    InversionRequest,
    InversionResponse,
)
from musak.core.notation.chord_serializer import inversion_to_score_data
from musak.core.schemas.common import FieldGroupSchema, FieldSchema
from musak.modules.chords.exporter import to_midi
from musak.modules.chords.generator import (
    generate_all_inversions,
    get_random_chord_inversion,
)
from musak.paths import INVERSIONS_CONFIG
from musak.shared.dict import namedtuple_with_base_note
from musak.shared.exporter import midi_to_audio
from musak.shared.files import load_yaml

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
    def _build_chord_info(chord_inversion: Any) -> dict[str, Any]:
        return namedtuple_with_base_note(chord_inversion)

    def generate(self, request: InversionRequest) -> InversionResponse:
        chords = self._resolve_chords(request)

        inversions = generate_all_inversions(dict(chords))
        chord_inversion = get_random_chord_inversion(
            inversions,
            lowest_note=request.lowest_note,
            highest_note=request.highest_note,
        )
        score_data = inversion_to_score_data(chord_inversion, sequential=request.sequential, tempo=request.tempo)

        midi_file = to_midi(
            chord_inversion.chord,
            tempo=request.tempo,
            sequential=request.sequential,
        )
        audio_data = midi_to_audio(midi_file)

        inversions_numbers = {chord_type: len(inv_list) for chord_type, inv_list in inversions.items()}

        return InversionResponse(
            audio_data=audio_data,
            score_data=score_data,
            chord_info=self._build_chord_info(chord_inversion),
            chord_types=list(chords.keys()),
            inversions_numbers=inversions_numbers,
        )
