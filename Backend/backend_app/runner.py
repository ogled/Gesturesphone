import threading
import os

import uvicorn

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

from . import state
from .app import app
from .utils.browser import open_in_browser


def run():
    def start_server():
        uvicorn.run(
            app,
            host=state.host,
            port=state.port,
            reload=False,
            log_level="warning",
            access_log=False,
        )

    server_thread = threading.Thread(target=start_server)
    server_thread.start()

    probe_host = "127.0.0.1" if state.host == "0.0.0.0" else state.host

    open_in_browser(
        url=f"http://{probe_host}:{state.port}",
        probe_url=f"http://{probe_host}:{state.port}/api/health",
    )

    try:
        server_thread.join()
    except KeyboardInterrupt:
        print("Shutting down...")
