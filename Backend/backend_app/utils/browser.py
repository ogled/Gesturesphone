import time

import requests
import webview


LOADING_HTML = """
<!doctype html>
<html lang="ru">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Gesturesphone</title>
    <style>
      :root {
        color-scheme: light;
      }
      body {
        margin: 0;
        min-height: 100vh;
        display: grid;
        place-content: center;
        gap: 14px;
        background: radial-gradient(circle at top right, #e0ecff, #eef2f7 55%);
        font-family: "Segoe UI", Tahoma, sans-serif;
        color: #111827;
      }
      .spinner {
        width: 42px;
        height: 42px;
        margin: 0 auto;
        border-radius: 50%;
        border: 4px solid #bfdbfe;
        border-top-color: #2563eb;
        animation: spin 0.8s linear infinite;
      }
      .title {
        font-size: 1.1rem;
        text-align: center;
        font-weight: 700;
      }
      .subtitle {
        color: #5b6776;
        text-align: center;
      }
      @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
      }
    </style>
  </head>
  <body>
    <div class="spinner"></div>
    <div class="title">Gesturesphone запускается</div>
    <div class="subtitle">Инициализация камеры, модели и AI...</div>
  </body>
</html>
"""

def _wait_server_and_open(window, url: str, probe_url: str):
    while True:
        try:
            resp = requests.get(probe_url, timeout=0.5)
            if resp.status_code == 200:
                window.load_url(url)
                return
        except requests.RequestException:
            pass
        time.sleep(0.2)


def open_in_browser(url: str, probe_url: str):
    window = webview.create_window("Gesturesphone", html=LOADING_HTML, resizable=True)
    webview.start(_wait_server_and_open, args=(window, url, probe_url))
    return False
