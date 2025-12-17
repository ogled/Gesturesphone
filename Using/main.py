import cv2
import mediapipe as mp
import threading
from flask import Flask, Response

# =============================
# Flask
# =============================
app = Flask(__name__)
latest_frame = None
frame_lock = threading.Lock()

def gen_frames():
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

def run_flask():
    app.run(host='0.0.0.0', port=5000, threaded=True)

# =============================
# MediaPipe Hands
# =============================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,          # ← САМОЕ БЫСТРОЕ
    min_detection_confidence=0.5,
    min_tracking_confidence=0.3
)

# =============================
# Camera
# =============================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)

# =============================
# Main loop
# =============================
threading.Thread(target=run_flask, daemon=True).start()

print("[INFO] Open http://<IP>:5000/video_feed")
frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if frame_id % 5 == 0:
        results = hands.process(rgb)
    frame_id += 1


    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_style.get_default_hand_landmarks_style(),
                mp_style.get_default_hand_connections_style()
            )

    with frame_lock:
        latest_frame = frame
    
    #cv2.imshow("Hands", frame)
    #if cv2.waitKey(1) == 27:
    #    break
cap.release()
hands.close()
cv2.destroyAllWindows()
