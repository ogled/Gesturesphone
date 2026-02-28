from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .lifespan import app_lifespan
from .routes.api import router as api_router
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

app = FastAPI(lifespan=app_lifespan)

app.mount(
    "/assets",
    StaticFiles(directory=BASE_DIR / "Frontend" / "dist" / "assets"),
    name="assets",
)

app.include_router(api_router)


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "Frontend" / "dist" / "index.html")