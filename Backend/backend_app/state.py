from collections import deque

host = "0.0.0.0"
port = 80
giga = None
text_from_ai = ""
ai_busy = False
temp_gestures_history = deque(maxlen=20)

clients = set()
state = {
    "status": "idle"
}
