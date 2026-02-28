import threading
import time

import requests
import uvicorn

from . import state
from .app import app
from .utils.browser import open_in_browser


def run():
    def start_server():
        uvicorn.run(app, host=state.host, port=state.port, reload=False)

    server_thread = threading.Thread(target=start_server)
    server_thread.start()

    probe_host = "127.0.0.1" if state.host == "0.0.0.0" else state.host

    server_up = False
    while not server_up:
        try:
            resp = requests.get(
                f"http://{probe_host}:{state.port}/api/getUsageVals", timeout=0.5
            )
            if resp.status_code == 200:
                server_up = True
        except requests.exceptions.RequestException:
            time.sleep(0.2)

    open_in_browser(f"http://{probe_host}:{state.port}")

    try:
        server_thread.join()
    except KeyboardInterrupt:
        print("Shutting down...")
