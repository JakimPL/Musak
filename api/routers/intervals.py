from fastapi import APIRouter, Request

from api.routers import form_str
from core.intervals.schema import IntervalConfigResponse, IntervalRequest, IntervalResponse
from core.intervals.service import IntervalService

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
        name: val
        for name, val in definitions.items()
        if form_str(form, f"interval_{name}") == "on"
    }

    req = IntervalRequest(
        tempo=int(form_str(form, "tempo", "120")),
        lowest_note=int(form_str(form, "lowest_note", "40")),
        highest_note=int(form_str(form, "highest_note", "90")),
        sequential=(form_str(form, "sequential") == "on"),
        intervals=intervals,
    )
    return _service.generate(req)
