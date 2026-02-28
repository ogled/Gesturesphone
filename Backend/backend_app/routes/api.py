import time

import cv2
import psutil
from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse

from .. import state
from ..core import picam as PiCam
from ..services.ai_tasks import run_ai_task, run_recording_ai_task

router = APIRouter(prefix="/api")


def gen_frames():
    while True:
        with PiCam.lock:
            frame = PiCam.latest_web_frame

        if frame is None:
            time.sleep(0.01)
            continue
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@router.get("/gestures")
def gestures_feed():
    data = PiCam.latest_gesture_probs or {}
    return JSONResponse(
        {
            "timestamp": data.get("ts", 0),
            "gestures": data.get("probs", {}),
        }
    )


@router.get("/camera")
def camera_feed():
    return StreamingResponse(
        gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/getUsageVals")
def cpu_usage_val():
    return JSONResponse(
        content={
            "CPU": str(round(psutil.cpu_percent(interval=0))),
            "RAM": str(round(psutil.virtual_memory().percent)),
            "FPS": str(round(PiCam.fps)),
        }
    )


@router.get("/start")
def api_start():
    if not PiCam.isAlreadyStart:
        PiCam.start()
    return JSONResponse({"status": "ok"})


@router.get("/gesture-history")
def get_gesture_history():
    if PiCam.recording_mode:
        return JSONResponse({"history": list()})
    return JSONResponse({"history": list(PiCam.gesture_history)})


@router.delete("/del-gesture")
def delete_gesture(id: int = 0, name: str = ""):
    if PiCam.gesture_history[id] == name:
        del PiCam.gesture_history[id]
    elif PiCam.gesture_history[id - 1] == name:
        del PiCam.gesture_history[id - 1]
    return JSONResponse({"status": "ok"})


@router.get("/ai-corect-gestures")
async def ai_corected_gestures(background_tasks: BackgroundTasks):
    if state.giga is None or state.ai_busy or len(PiCam.gesture_history) == 0:
        return {"status": "error"}
    state.temp_gestures_history = PiCam.gesture_history.copy()
    PiCam.gesture_history.clear()
    background_tasks.add_task(run_ai_task)
    return {"status": "started"}


@router.get("/ai-get-status")
async def ai_get_status():
    if state.giga is None:
        return {"status": "not_ready"}
    return {"status": "ready"}


@router.get("/clearHistory")
async def clear_history():
    PiCam.gesture_history.clear()
    return {"status": "ok"}


@router.get("/start-recording-mode")
def start_recording():
    PiCam.gesture_history.clear()
    PiCam.recording_mode = True
    return {"status": "recording_started"}


@router.get("/end-recording-mode")
def end_recording(background_tasks: BackgroundTasks):
    PiCam.recording_mode = False

    if not PiCam.recording_buffer:
        return {"status": "empty"}

    background_tasks.add_task(run_recording_ai_task)
    PiCam.gesture_history.clear()
    return {"status": "processing"}


@router.get("/get-ai-corect-text")
async def get_ai_corected_text():
    return JSONResponse({"text": state.text_from_ai, "busy": state.ai_busy})
