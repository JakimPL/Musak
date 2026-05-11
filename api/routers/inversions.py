from fastapi import APIRouter, Request

from api.routers import form_str
from config.defaults import HIGHEST_NOTE, LOWEST_NOTE, TEMPO
from core.inversions.schema import (
    InversionConfigResponse,
    InversionRequest,
    InversionResponse,
)
from core.inversions.service import InversionService

router = APIRouter()
_service = InversionService()


@router.get("/config", response_model=InversionConfigResponse)
async def get_config() -> InversionConfigResponse:
    return _service.get_config()


@router.post("/submit", response_model=InversionResponse)
async def submit(request: Request) -> InversionResponse:
    form = await request.form()
    definitions = _service.definitions

    chords = {name: intervals for name, intervals in definitions.items() if form_str(form, f"chord_{name}") == "on"}

    inversion_request = InversionRequest(
        tempo=int(form_str(form, "tempo", TEMPO)),
        lowest_note=int(form_str(form, "lowest_note", LOWEST_NOTE)),
        highest_note=int(form_str(form, "highest_note", HIGHEST_NOTE)),
        sequential=(form_str(form, "sequential") == "on"),
        chords=chords,
    )

    return _service.generate(inversion_request)
