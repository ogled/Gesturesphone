import os
import json
import shutil
import socket
import time
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "Config" / "config.json"
CONFIG_EXAMPLE_PATH = BASE_DIR / "Config" / "config.example.json"

system_prompt = ""
system_prompt_recording = ""



def _resolve_path(path_value):
    path = Path(path_value)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path

def internet_available(timeout=2):
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=timeout)
        return True
    except OSError:
        return False
    
def initialization():
    global system_prompt, system_prompt_recording

    if not CONFIG_PATH.exists():
        shutil.copy(CONFIG_EXAMPLE_PATH, CONFIG_PATH)
        print("config.json created. Please insert your API token.")
        return None

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    token = config.get("TOKEN", "").strip()

    if not token:
        print("API TOKEN is empty")
        return None

    system_prompt_path = config.get("PATH_TO_SYSTEM_PROMT", "")
    resolved_system_prompt_path = _resolve_path(system_prompt_path) if system_prompt_path else None

    if resolved_system_prompt_path and resolved_system_prompt_path.exists():
        with open(resolved_system_prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    else:
        print("System prompt not found")
        return None

    system_prompt_path = config.get("PATH_TO_SYSTEM_PROMT_RECORDING", "")
    resolved_recording_prompt_path = _resolve_path(system_prompt_path) if system_prompt_path else None

    if resolved_recording_prompt_path and resolved_recording_prompt_path.exists():
        with open(resolved_recording_prompt_path, "r", encoding="utf-8") as f:
            system_prompt_recording = f.read()
    else:
        print("System prompt recording not found")
        return None
    
    if not internet_available():
        print("No internet connection")
        return None

    giga = GigaChat(
        credentials=token,
        verify_ssl_certs=False,
        model="GigaChat-2",
        timeout=4
    )

    print("GigaChat initialized")

    return giga

def get_response(giga, user_text, retries=3, retry_delay=2):
    global system_prompt

    if not internet_available():
        return "[ERROR] No internet connection"

    payload = Chat(
        messages=[
            Messages(
                role=MessagesRole.SYSTEM,
                content=system_prompt
            ),
            Messages(
                role=MessagesRole.USER,
                content=user_text
            )
        ]
    )

    for attempt in range(retries):
        try:
            response = giga.chat(payload)
            return response.choices[0].message.content

        except Exception as e:
            print(f"Request failed ({attempt + 1}/{retries}): {e}")

            if attempt + 1 == retries:
                return "[ERROR] GigaChat service unavailable"

            time.sleep(retry_delay)

def get_response_recording_mode(giga, user_text, retries=3, retry_delay=2):
    global system_prompt_recording

    if not internet_available():
        return "[ERROR] No internet connection"
    payload = Chat(
        messages=[
            Messages(
                role=MessagesRole.SYSTEM,
                content=system_prompt_recording
            ),
            Messages(
                role=MessagesRole.USER,
                content=user_text
            )
        ]
    )

    for attempt in range(retries):
        try:
            response = giga.chat(payload)
            return response.choices[0].message.content

        except Exception as e:
            print(f"Request failed ({attempt + 1}/{retries}): {e}")

            if attempt + 1 == retries:
                return "[ERROR] GigaChat service unavailable"

            time.sleep(retry_delay)

def close(giga):
    if giga:
        try:
            giga.close()
        except:
            pass

if __name__ == "__main__":
    giga = initialization()

    if giga:
        result = get_response(
            giga,
            "Я делать вещь который помочь люди который плохо слышать"
        )
        print(result)
        close(giga)