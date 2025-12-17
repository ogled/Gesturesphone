from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import json

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

@app.get("/")
def index():
    return FileResponse("../Frontend/dist/index.html")

@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    clients.add(ws)

    # отправляем текущее состояние при подключении
    await ws.send_text(json.dumps(state))

    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)

            if data["cmd"] == "start":
                state["status"] = "running"
            elif data["cmd"] == "stop":
                state["status"] = "stopped"

            # рассылаем ВСЕМ
            for c in clients:
                await c.send_text(json.dumps(state))

    except:
        clients.remove(ws)
