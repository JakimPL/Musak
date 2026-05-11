from fastapi import APIRouter, Request

from musak.api.routers import form_str
from musak.config.defaults import (
    GROUPS,
    MEASURES,
    TEMPO,
    TIME_SIGNATURE_DENOMINATOR,
    TIME_SIGNATURE_NUMERATOR,
)
from musak.core.rhythm.schema import RhythmConfigResponse, RhythmRequest, RhythmResponse
from musak.core.rhythm.service import RhythmService, note_map, phrase_map

router = APIRouter()
_service = RhythmService()

_NOTE_KEYS = [
    "whole_note",
    "half_note",
    "quarter_note",
    "eighth_note",
    "sixteenth_note",
    "thirty_second_note",
    "whole_rest",
    "half_rest",
    "quarter_rest",
    "eighth_rest",
    "sixteenth_rest",
    "thirty_second_rest",
    "dotted_half_note",
    "dotted_quarter_note",
    "dotted_eighth_note",
    "dotted_sixteenth_note",
]
_PHRASE_KEYS = [
    "two_quarter_notes_phrase",
    "two_eighth_notes_phrase",
    "four_eighth_notes_phrase",
    "two_sixteenth_notes_phrase",
    "four_sixteenth_notes_phrase",
    "eight_sixteenth_notes_phrase",
    "left_quarter_phrase",
    "right_quarter_phrase",
    "left_eighth_phrase",
    "right_eighth_phrase",
    "left_sixteenth_phrase",
    "right_sixteenth_phrase",
]


@router.get("/config", response_model=RhythmConfigResponse)
async def get_config() -> RhythmConfigResponse:
    return _service.get_config()


@router.post("/submit", response_model=RhythmResponse)
async def submit(request: Request) -> RhythmResponse:
    form = await request.form()

    notes = [note_map[key] for key in _NOTE_KEYS if form_str(form, key) == "on"]
    phrases = [phrase_map[key] for key in _PHRASE_KEYS if form_str(form, key) == "on"]

    rhythm_request = RhythmRequest(
        tempo=int(form_str(form, "tempo", TEMPO)),
        groups=int(form_str(form, "groups", GROUPS)),
        measures=int(form_str(form, "measures", MEASURES)),
        time_signature=(
            int(
                form_str(
                    form,
                    "time_signature_numerator",
                    TIME_SIGNATURE_NUMERATOR,
                )
            ),
            int(
                form_str(
                    form,
                    "time_signature_denominator",
                    TIME_SIGNATURE_DENOMINATOR,
                )
            ),
        ),
        notes=notes,
        phrases=phrases,
        custom_phrases=form_str(form, "custom_phrases"),
    )

    return _service.generate(rhythm_request)
