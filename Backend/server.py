from collections import deque
import threading
from fastapi import BackgroundTasks, FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
import AiTextCorecting
from contextlib import asynccontextmanager
import psutil
import os
import cv2
import time
import PiCam
import uvicorn
import requests
import shutil
import subprocess

host = "0.0.0.0"
giga = None
port = 80
textFromAi = ""
ai_busy = False
tempGesturesHistory = deque(maxlen = 20)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global giga
    PiCam.initialization()
    giga = AiTextCorecting.initialization()
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
    if PiCam.recording_mode:
        return JSONResponse({
            "history": list()
        })
    else:
        return JSONResponse({
            "history": list(PiCam.gesture_history)
        })
    
@app.delete("/api/del-gesture")
def delete_gesture(id: int = 0, name: str = ""):
    if PiCam.gesture_history[id] == name:
        del PiCam.gesture_history[id]
    elif PiCam.gesture_history[id - 1] == name:
         del PiCam.gesture_history[id - 1]
    return JSONResponse({"status": "ok"})

@app.get("/")
def index():
    return FileResponse("../Frontend/dist/index.html")

def run_ai_task():
    global textFromAi, ai_busy, tempGesturesHistory

    ai_busy = True

    try:
        text = ' '.join(tempGesturesHistory)
        textFromAi = AiTextCorecting.get_response(giga, text)

    except Exception as e:
        textFromAi = "[ERROR] AI unavailable"
        print("AI error:", e)

    ai_busy = False

def run_recording_ai_task():
    global textFromAi, ai_busy, recording_buffer

    ai_busy = True

    try:
        lines = []
        for g in PiCam.recording_buffer:
            lines.append(f"{g['gesture']} ({g['duration']} сек, {g['confidence']}%)")

        payload_text = "".join(lines)

        textFromAi = AiTextCorecting.get_response_recording_mode(giga, user_text=payload_text)
        PiCam.gesture_history.clear()
        PiCam.gesture_history.append(textFromAi)

    except Exception as e:
        textFromAi = "[ERROR] AI unavailable"
        print("AI error:", e)

    finally:
        PiCam.recording_buffer.clear()
        ai_busy = False

@app.get("/api/ai-corect-gestures")
async def ai_corected_gestures(background_tasks: BackgroundTasks):
    global tempGesturesHistory, giga, ai_busy
    if giga is None or ai_busy or len(PiCam.gesture_history) == 0:
        return {"status": "error"}
    tempGesturesHistory = PiCam.gesture_history.copy()
    PiCam.gesture_history.clear()
    background_tasks.add_task(run_ai_task)
    return {"status": "started"}

@app.get("/api/ai-get-status")
async def ai_get_status():
    global tempGesturesHistory, giga, ai_busy
    if giga is None:
        return {"status": "not_ready"}
    return {"status": "ready"}

@app.get("/api/clearHistory")
async def clear_history():
    PiCam.gesture_history.clear()
    return {"status": "ok"}

@app.get("/api/start-recording-mode")
def start_recording():
    PiCam.gesture_history.clear()
    PiCam.recording_mode = True
    return {"status": "recording_started"}


@app.get("/api/end-recording-mode")
def end_recording(background_tasks: BackgroundTasks):
    PiCam.recording_mode = False

    if not PiCam.recording_buffer:
        return {"status": "empty"}

    background_tasks.add_task(run_recording_ai_task)
    PiCam.gesture_history.clear()
    return {"status": "processing"}

@app.get("/api/get-ai-corect-text")
async def get_ai_corected_text():
    global textFromAi, ai_busy
    return JSONResponse({
        "text": textFromAi,
        "busy": ai_busy
    })

def open_in_browser(url):
    if "DISPLAY" not in os.environ:
        print("[WARN] No DISPLAY found, GUI not available")
        return False

    browsers = [
        ("chromium-browser", ["--disable-infobars", "--noerrdialogs", "--window-focus", "--start-fullscreen", "--password-store=basic", "--user-data-dir=/tmp/chrome_temp_profile"]),
        ("chromium", ["--disable-infobars", "--noerrdialogs", "--window-focus", "--start-fullscreen", "--password-store=basic", "--user-data-dir=/tmp/chrome_temp_profile"]),
    ]

    for browser, args in browsers:
        path = shutil.which(browser)
        if path:
            subprocess.Popen(
                [path, *args, url],
                env=os.environ
            )
            print(f"[INFO] Opened in {browser}")
            return True

    print("[WARN] Chromium not found")
    return False

if __name__ == "__main__":
    def start_server():
        uvicorn.run("server:app", host=host, port=port, reload=False)

    server_thread = threading.Thread(target=start_server)
    server_thread.start()

    if host == "0.0.0.0":
        host = "127.0.0.1"

    server_up = False
    while not server_up:
        try:
            resp = requests.get(f"http://{host}:{port}/api/getUsageVals", timeout=0.5)
            if resp.status_code == 200:
                server_up = True
        except requests.exceptions.RequestException:
            time.sleep(0.2)
    
    open_in_browser(f"http://{host}:{port}")

    try:
        server_thread.join()
    except KeyboardInterrupt:
        print("Shutting down...")