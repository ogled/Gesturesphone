from .. import state
from ..core import ai_text_correcting as AiTextCorecting
from ..core import picam as PiCam


def run_ai_task():
    state.ai_busy = True

    try:
        text = " ".join(state.temp_gestures_history)
        state.text_from_ai = AiTextCorecting.get_response(state.giga, text)
    except Exception as e:
        state.text_from_ai = "[ERROR] AI unavailable"
        print("AI error:", e)

    state.ai_busy = False


def run_recording_ai_task():
    state.ai_busy = True

    try:
        lines = []
        for g in PiCam.recording_buffer:
            lines.append(f"{g['gesture']} ({g['duration']} сек, {g['confidence']}%)")

        payload_text = "".join(lines)

        state.text_from_ai = AiTextCorecting.get_response_recording_mode(
            state.giga,
            user_text=payload_text,
        )
        PiCam.gesture_history.clear()
        PiCam.gesture_history.append(state.text_from_ai)

    except Exception as e:
        state.text_from_ai = "[ERROR] AI unavailable"
        print("AI error:", e)

    finally:
        PiCam.recording_buffer.clear()
        state.ai_busy = False
