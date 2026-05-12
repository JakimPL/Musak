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
from musak.core.rhythm.service import NOTE_KEYS, PHRASE_KEYS, RhythmService, note_map, phrase_map

router = APIRouter()
_service = RhythmService()


@router.get("/config", response_model=RhythmConfigResponse)
async def get_config() -> RhythmConfigResponse:
    return _service.get_config()


@router.post("/submit", response_model=RhythmResponse)
async def submit(request: Request) -> RhythmResponse:
    form = await request.form()

    notes = [note_map[key] for key in NOTE_KEYS if form_str(form, key) == "on"]
    phrases = [phrase_map[key] for key in PHRASE_KEYS if form_str(form, key) == "on"]

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
        melodic=form_str(form, "melodic") == "on",
        notes=notes,
        phrases=phrases,
        custom_phrases=form_str(form, "custom_phrases"),
    )

    return _service.generate(rhythm_request)
