from contextlib import asynccontextmanager

from . import state
from .core import ai_text_correcting as AiTextCorecting
from .core import picam as PiCam


@asynccontextmanager
async def app_lifespan(app):
    PiCam.initialization()
    state.giga = AiTextCorecting.initialization()
    yield
