import re
from typing import Any

from pydantic import ValidationError

from musak.config.defaults import (
    GROUPS,
    MAX_GROUPS,
    MAX_MEASURES,
    MAX_TEMPO,
    MAX_TIME_SIGNATURE_NUMERATOR,
    MEASURES,
    MELODIC,
    MIN_GROUPS,
    MIN_MEASURES,
    MIN_TEMPO,
    MIN_TIME_SIGNATURE_NUMERATOR,
    TEMPO,
    TIME_SIGNATURE_DENOMINATOR,
    TIME_SIGNATURE_DENOMINATOR_OPTIONS,
    TIME_SIGNATURE_NUMERATOR,
)
from musak.config.models import RhythmConfig
from musak.core.notation.rhythm_serializer import phrases_to_score_data
from musak.core.rhythm.schema import (
    NoteValue,
    RhythmConfigResponse,
    RhythmRequest,
    RhythmResponse,
)
from musak.core.schemas.common import FieldGroupSchema, FieldSchema
from musak.modules.elements.note import Note
from musak.modules.elements.phrase import Phrase
from musak.modules.rhythm.conversion import phrases_to_midi
from musak.modules.rhythm.exceptions import RhygenException
from musak.modules.rhythm.generator import RhythmGenerator
from musak.modules.rhythm.settings import GroupSettings, Settings
from musak.paths import RHYTHM_CONFIG
from musak.shared.files import load_yaml
from musak_shared.exporter import midi_to_audio

note_map: dict[str, NoteValue] = {
    "whole_note": 1,
    "half_note": 2,
    "quarter_note": 4,
    "eighth_note": 8,
    "sixteenth_note": 16,
    "thirty_second_note": 32,
    "whole_rest": -1,
    "half_rest": -2,
    "quarter_rest": -4,
    "eighth_rest": -8,
    "sixteenth_rest": -16,
    "thirty_second_rest": -32,
    "dotted_half_note": (3, 4),
    "dotted_quarter_note": (3, 8),
    "dotted_eighth_note": (3, 16),
    "dotted_sixteenth_note": (3, 32),
}

phrase_map: dict[str, list[NoteValue]] = {
    "two_quarter_notes_phrase": [4, 4],
    "two_eighth_notes_phrase": [8, 8],
    "four_eighth_notes_phrase": [8, 8, 8, 8],
    "two_sixteenth_notes_phrase": [16, 16],
    "four_sixteenth_notes_phrase": [16, 16, 16, 16],
    "eight_sixteenth_notes_phrase": [16, 16, 16, 16, 16, 16, 16, 16],
    "left_quarter_phrase": [4, -4],
    "right_quarter_phrase": [-4, 4],
    "left_eighth_phrase": [8, -8],
    "right_eighth_phrase": [-8, 8],
    "left_sixteenth_phrase": [16, -16],
    "right_sixteenth_phrase": [-16, 16],
}

settings_map: dict[str, NoteValue | list[NoteValue]] = {**note_map, **phrase_map}

NOTE_KEYS: tuple[str, ...] = tuple(note_map)
PHRASE_KEYS: tuple[str, ...] = tuple(phrase_map)

_NOTE_LABELS = {
    "whole_note": "\U0001d15d",
    "half_note": "\U0001d15e",
    "quarter_note": "\U0001d15f",
    "eighth_note": "\U0001d160",
    "sixteenth_note": "\U0001d160",
    "thirty_second_note": "\U0001d161",
    "whole_rest": "\U0001d13b",
    "half_rest": "\U0001d13c",
    "quarter_rest": "\U0001d13d",
    "eighth_rest": "\U0001d13e",
    "sixteenth_rest": "\U0001d13f",
    "thirty_second_rest": "\U0001d140",
    "dotted_half_note": "\U0001d15e.",
    "dotted_quarter_note": "\U0001d15f.",
    "dotted_eighth_note": "\U0001d160.",
    "dotted_sixteenth_note": "\U0001d160.",
    "two_quarter_notes_phrase": "\U0001d15f\U0001d15f",
    "two_eighth_notes_phrase": "\U0000266b",
    "four_eighth_notes_phrase": "\U0000266b\U0000266b",
    "two_sixteenth_notes_phrase": "\U0000266c",
    "four_sixteenth_notes_phrase": "\U0000266c\U0000266c",
    "eight_sixteenth_notes_phrase": "\U0000266c\U0000266c\U0000266c\U0000266c",
    "left_quarter_phrase": "\U0001d15f\u00a0\U0001d13d",
    "right_quarter_phrase": "\U0001d13d\u00a0\U0001d15f",
    "left_eighth_phrase": "\U0001d160\u00a0\U0001d13e",
    "right_eighth_phrase": "\U0001d13e\u00a0\U0001d160",
    "left_sixteenth_phrase": "\U0001d161\u00a0\U0001d13f",
    "right_sixteenth_phrase": "\U0001d13f\u00a0\U0001d161",
}


def parse_custom_phrase(raw_phrase: str) -> list[int | tuple[int, int]]:
    elements = raw_phrase.split(",")
    notes: list[int | tuple[int, int]] = []

    for element in elements:
        if "(" in element or ")" in element:
            if re.match(r"(-)?\(\d+:\d+\)", element):
                raw_pair = element.replace("(", "").replace(")", "").split(":")
                notes.append((int(raw_pair[0]), int(raw_pair[1])))
            else:
                raise ValueError(f"invalid note element {element}")
        else:
            notes.append(int(element))

    return notes


def parse_custom_phrases(phrases_string: str) -> list[list[int | tuple[int, int]]]:
    if not phrases_string:
        return []

    string = phrases_string.strip().replace(" ", "")

    count = 0
    elements = []
    raw_phrase = ""
    for char in string:
        if char not in [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            ",",
            "[",
            "]",
            "(",
            ")",
            "-",
            ":",
        ]:
            raise ValueError(f"unexpected symbol {char}")

        bracket = True
        if char == "[":
            count += 1
        elif char == "]":
            count -= 1
            if raw_phrase:
                elements.append(raw_phrase)
            raw_phrase = ""
        else:
            bracket = False
            if count != 0:
                raw_phrase += char

        if count < 0:
            raise ValueError("unexpected closing bracket")

        if count > 1:
            raise ValueError("unexpected nested expression")

        if count == 0 and char != "," and not bracket:
            raise ValueError(f"unexpected symbol {char} outside the expression")

    if count != 0:
        raise ValueError("unbalanced square brackets")

    return [parse_custom_phrase(raw_phrase) for raw_phrase in elements]


def _load_config() -> RhythmConfig:
    return load_yaml(RHYTHM_CONFIG, RhythmConfig)


def _load_defaults() -> dict[str, Any]:
    return _load_config().default_settings.model_dump()


class RhythmService:
    def get_config(self) -> RhythmConfigResponse:
        defaults = _load_defaults()

        tempo_group = FieldGroupSchema(
            label="Tempo",
            fields=[
                FieldSchema(
                    name="melodic",
                    type="boolean",
                    label="Melodic (piano)",
                    default=MELODIC,
                ),
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

        structure_group = FieldGroupSchema(
            label="Structure",
            fields=[
                FieldSchema(
                    name="groups",
                    type="slider",
                    label="Groups",
                    default=defaults.get("groups", GROUPS),
                    min=MIN_GROUPS,
                    max=MAX_GROUPS,
                ),
                FieldSchema(
                    name="measures",
                    type="slider",
                    label="Measures",
                    default=defaults.get("measures", MEASURES),
                    min=MIN_MEASURES,
                    max=MAX_MEASURES,
                ),
            ],
        )

        time_signature_group = FieldGroupSchema(
            label="Time signature",
            fields=[
                FieldSchema(
                    name="time_signature_numerator",
                    type="slider",
                    label="Numerator",
                    default=defaults.get("time_signature_numerator", TIME_SIGNATURE_NUMERATOR),
                    min=MIN_TIME_SIGNATURE_NUMERATOR,
                    max=MAX_TIME_SIGNATURE_NUMERATOR,
                ),
                FieldSchema(
                    name="time_signature_denominator",
                    type="slider",
                    label="Denominator",
                    default=defaults.get("time_signature_denominator", TIME_SIGNATURE_DENOMINATOR),
                    options=list(TIME_SIGNATURE_DENOMINATOR_OPTIONS),
                ),
            ],
        )

        def _bool_field(name: str) -> FieldSchema:
            return FieldSchema(
                name=name,
                type="boolean",
                label=_NOTE_LABELS[name],
                default=(defaults.get(name) == "on"),
            )

        notes_group = FieldGroupSchema(
            label="Notes",
            fields=[
                _bool_field("whole_note"),
                _bool_field("half_note"),
                _bool_field("quarter_note"),
                _bool_field("eighth_note"),
                _bool_field("sixteenth_note"),
                _bool_field("thirty_second_note"),
            ],
        )

        rests_group = FieldGroupSchema(
            label="Rests",
            fields=[
                _bool_field("whole_rest"),
                _bool_field("half_rest"),
                _bool_field("quarter_rest"),
                _bool_field("eighth_rest"),
                _bool_field("sixteenth_rest"),
                _bool_field("thirty_second_rest"),
            ],
        )

        dotted_group = FieldGroupSchema(
            label="Dotted notes",
            fields=[
                _bool_field("dotted_half_note"),
                _bool_field("dotted_quarter_note"),
                _bool_field("dotted_eighth_note"),
                _bool_field("dotted_sixteenth_note"),
            ],
        )

        phrases_group = FieldGroupSchema(
            label="Phrases",
            fields=[
                _bool_field("two_quarter_notes_phrase"),
                _bool_field("two_eighth_notes_phrase"),
                _bool_field("four_eighth_notes_phrase"),
                _bool_field("two_sixteenth_notes_phrase"),
                _bool_field("four_sixteenth_notes_phrase"),
                _bool_field("eight_sixteenth_notes_phrase"),
            ],
        )

        syncope_group = FieldGroupSchema(
            label="Syncope",
            fields=[
                _bool_field("left_quarter_phrase"),
                _bool_field("right_quarter_phrase"),
                _bool_field("left_eighth_phrase"),
                _bool_field("right_eighth_phrase"),
                _bool_field("left_sixteenth_phrase"),
                _bool_field("right_sixteenth_phrase"),
            ],
        )

        custom_phrases_group = FieldGroupSchema(
            label="Custom phrases",
            fields=[
                FieldSchema(
                    name="custom_phrases",
                    type="text",
                    label="",
                    default="",
                    placeholder="[4,8,8][-4,4]",
                    tooltip=(
                        "Each [...] is one phrase.\n"
                        "Numbers are note denominators: 1=whole, 2=half, 4=quarter, etc.\n"
                        "Negative values are rests (e.g. -4 = quarter rest).\n"
                        "Dotted notes use (numerator:denominator) syntax, e.g. (3:8) = dotted quarter."
                    ),
                ),
            ],
        )

        return RhythmConfigResponse(
            groups=[
                tempo_group,
                structure_group,
                time_signature_group,
                notes_group,
                rests_group,
                dotted_group,
                phrases_group,
                syncope_group,
                custom_phrases_group,
            ]
        )

    def _build_settings(self, request: RhythmRequest) -> Settings:
        config = _load_config()
        notes, phrases = self._collect_notes_phrases(request)
        if notes or phrases:
            default_group_settings = GroupSettings.model_validate({"notes": notes, "phrases": phrases})
        else:
            default_group_settings = GroupSettings.model_validate(config.default_group.model_dump())

        return Settings(
            tempo=request.tempo,
            groups=request.groups,
            measures=request.measures,
            time_signature=request.time_signature,
            default_group_settings=default_group_settings,
        )

    @staticmethod
    def _is_time_signature_error(exception: ValidationError) -> bool:
        return any(error["loc"] == ("time_signature",) for error in exception.errors())

    def _collect_notes_phrases(self, request: RhythmRequest) -> tuple[list[Note], list[Phrase]]:
        notes = [Note.model_validate(note) for note in request.notes]
        phrases_raw = list(request.phrases) + parse_custom_phrases(request.custom_phrases)
        phrases = [Phrase.model_validate(phrase) for phrase in phrases_raw]
        return notes, phrases

    def generate(self, request: RhythmRequest) -> RhythmResponse:
        try:
            settings = self._build_settings(request)
        except ValidationError as exception:
            return RhythmResponse(
                exception=str(exception),
                time_signature_error=self._is_time_signature_error(exception),
            )
        except ValueError as exception:
            return RhythmResponse(exception=str(exception))

        try:
            generator = RhythmGenerator(settings)
            phrase_list = generator()
        except RhygenException as exception:
            return RhythmResponse(
                exception=str(exception),
                time_signature_error=False,
            )

        score_data = phrases_to_score_data(
            phrase_list,
            time_signature=settings.time_signature,
            tempo=settings.tempo,
            max_notes_per_measure=generator.max_notes_per_measure,
            melodic=request.melodic,
        )

        midi_file = phrases_to_midi(
            phrase_list,
            time_signature=settings.time_signature,
            tempo=settings.tempo,
            melodic=request.melodic,
        )
        audio_data = midi_to_audio(midi_file)

        return RhythmResponse(
            audio_data=audio_data,
            score_data=score_data,
            exception=None,
            time_signature_error=False,
        )
