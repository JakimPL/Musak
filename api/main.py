import os

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from api.routers import intervals, inversions, rhythm

os.makedirs("temp", exist_ok=True)

app = FastAPI(title="Musak")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/temp", StaticFiles(directory="temp"), name="temp")

templates = Jinja2Templates(directory="templates")

app.include_router(intervals.router, prefix="/api/intervals", tags=["intervals"])
app.include_router(inversions.router, prefix="/api/inversions", tags=["inversions"])
app.include_router(rhythm.router, prefix="/api/rhythm", tags=["rhythm"])


@app.get("/")
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")


@app.get("/intervals/")
async def intervals_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "intervals.html")


@app.get("/inversions/")
async def inversions_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "inversions.html")


@app.get("/rhythm/")
async def rhythm_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "rhythm.html")
