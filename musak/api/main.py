import os

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from musak.api.routers import intervals, inversions, rhythm

DEBUG = os.getenv("DEBUG", "0") == "1"

app = FastAPI(title="Musak", debug=DEBUG)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

app.include_router(intervals.router, prefix="/api/intervals", tags=["intervals"])
app.include_router(inversions.router, prefix="/api/inversions", tags=["inversions"])
app.include_router(rhythm.router, prefix="/api/rhythm", tags=["rhythm"])


@app.get("/")
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", {"active_page": None})


@app.get("/intervals/")
async def intervals_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "intervals.html", {"debug": DEBUG, "active_page": "intervals"})


@app.get("/inversions/")
async def inversions_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "inversions.html", {"debug": DEBUG, "active_page": "inversions"})


@app.get("/rhythm/")
async def rhythm_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "rhythm.html", {"debug": DEBUG, "active_page": "rhythm"})
