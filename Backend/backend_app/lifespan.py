from contextlib import asynccontextmanager

import AiTextCorecting
import PiCam

from . import state


@asynccontextmanager
async def app_lifespan(app):
    PiCam.initialization()
    state.giga = AiTextCorecting.initialization()
    yield
