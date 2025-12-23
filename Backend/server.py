from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
import psutil
import json
import cv2
import os
import time
import atexit
import signal
import sys
import PiCam

GESTURES_FILE = "gestures.json"
camera = cv2.VideoCapture(0)
app = FastAPI()

clients = set()
state = {
    "status": "idle"
}

app.mount(
    "/assets",
    StaticFiles(directory="../Frontend/dist/assets"),
    name="assets"
)

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
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )


PiCam.initialization()

@app.get("/api/gestures")
def gestures_feed():
    data = PiCam.latest_gesture_probs or {}
    return JSONResponse({
        "timestamp": data.get("ts", 0),
        "gestures": data.get("probs", {})
    })


@app.get("/api/camera")
def camera_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/getUsageVals")
def cpu_usage_val():
    return JSONResponse(
        content={"CPU": str(round(psutil.cpu_percent(interval=0))), "RAM": str(round(psutil.virtual_memory().percent))}
    )

@app.get("/api/start")
def api_start():
    if not PiCam.isAlreadyStart:
        PiCam.start()
    return JSONResponse({"status": "ok"})

@app.get("/")
def index():
    return FileResponse("../Frontend/dist/index.html")

@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    clients.add(ws)

    await ws.send_text(json.dumps(state))

    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)

            if data["cmd"] == "start":
                state["status"] = "running"
            elif data["cmd"] == "stop":
                state["status"] = "stopped"
            for c in clients:
                await c.send_text(json.dumps(state))

    except:
        clients.remove(ws)

def cleanup_on_exit(signum, frame):
    if os.path.exists(GESTURES_FILE):
        os.remove(GESTURES_FILE)
    if PiCam.cap and PiCam.cap.isOpened():
        PiCam.cap.release()
    if PiCam.out:
        PiCam.out.release()
    if PiCam.hands:
        PiCam.hands.close()
    sys.exit(0)

#signal.signal(signal.SIGINT, cleanup_on_exit)
#signal.signal(signal.SIGTERM, cleanup_on_exit)
