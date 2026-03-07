from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .lifespan import app_lifespan
from .routes.api import router as api_router
from pathlib import Path
import sys


def _iter_runtime_roots():
    module_path = Path(__file__).resolve()
    yield module_path.parents[2]

    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            yield Path(meipass)
        exe_dir = Path(sys.executable).resolve().parent
        yield exe_dir
        yield exe_dir / "_internal"


def _resolve_frontend_dist() -> Path:
    checked = []
    seen = set()
    for root in _iter_runtime_roots():
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        candidate = root / "Frontend" / "dist"
        checked.append(str(candidate))
        if candidate.exists():
            return candidate

    raise RuntimeError(
        "Frontend dist directory was not found. Checked: " + "; ".join(checked)
    )


FRONTEND_DIST_DIR = _resolve_frontend_dist()
ASSETS_DIR = FRONTEND_DIST_DIR / "assets"

app = FastAPI(lifespan=app_lifespan)

app.mount(
    "/assets",
    StaticFiles(directory=ASSETS_DIR),
    name="assets",
)

app.include_router(api_router)


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIST_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon_ico():
    return FileResponse(FRONTEND_DIST_DIR / "favicon.ico")


@app.get("/favicon.svg", include_in_schema=False)
def favicon_svg():
    return FileResponse(FRONTEND_DIST_DIR / "favicon.svg")
