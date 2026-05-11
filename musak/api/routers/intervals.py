from fastapi import APIRouter, Request

from musak.api.routers import form_str
from musak.config.defaults import HIGHEST_NOTE, LOWEST_NOTE, TEMPO
from musak.core.intervals.schema import (
    IntervalConfigResponse,
    IntervalRequest,
    IntervalResponse,
)
from musak.core.intervals.service import IntervalService

router = APIRouter()
_service = IntervalService()


@router.get("/config", response_model=IntervalConfigResponse)
async def get_config() -> IntervalConfigResponse:
    return _service.get_config()


@router.post("/submit", response_model=IntervalResponse)
async def submit(request: Request) -> IntervalResponse:
    form = await request.form()
    definitions = _service.definitions

    intervals = {
        name: semitones for name, semitones in definitions.items() if form_str(form, f"interval_{name}") == "on"
    }

    interval_request = IntervalRequest(
        tempo=int(form_str(form, "tempo", TEMPO)),
        lowest_note=int(form_str(form, "lowest_note", LOWEST_NOTE)),
        highest_note=int(form_str(form, "highest_note", HIGHEST_NOTE)),
        sequential=(form_str(form, "sequential") == "on"),
        intervals=intervals,
    )

    return _service.generate(interval_request)
