import threading
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from contextlib import asynccontextmanager
import psutil
import json
import cv2
import time
import PiCam
import uvicorn
import webview
import requests
import platform

host = "0.0.0.0"
port = 80

@asynccontextmanager
async def lifespan(app: FastAPI):
    PiCam.initialization()
    yield

app = FastAPI(lifespan=lifespan)


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
        content={"CPU": str(round(psutil.cpu_percent(interval=0))), "RAM": str(round(psutil.virtual_memory().percent)), "FPS": str(round(PiCam.fps))}
    )

@app.get("/api/start")
def api_start():
    if not PiCam.isAlreadyStart:
        PiCam.start()
    return JSONResponse({"status": "ok"})

@app.get("/api/gesture-history")
def get_gesture_history():
    return JSONResponse({
        "history": list(PiCam.gesture_history)
    })

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

if __name__ == "__main__":
    def start_server():
        uvicorn.run("server:app", host=host, port=port, reload=False)

    threading.Thread(target=start_server, daemon=True).start()
    if host == "0.0.0.0":
        host = "127.0.0.1"

    scale = 1.0
    if platform.system() == "Linux":
        scale = 0.8 

    server_up = False
    while not server_up:
        try:
            resp = requests.get(f"http://{host}:{port}/api/getUsageVals", timeout=0.5)
            if resp.status_code == 200:
                server_up = True
        except requests.exceptions.RequestException:
            time.sleep(0.2)
    
    window = webview.create_window("Gesturesphone", f"http://{host}:{port}", fullscreen=True, focus=True, resizable=True)
    webview.start(func=lambda: window.evaluate_js(f"document.body.style.zoom='{scale*100}%'"))