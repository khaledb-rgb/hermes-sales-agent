import os

import requests
from dotenv import load_dotenv

load_dotenv()

_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
_BASE_URL = f"https://api.telegram.org/bot{_TOKEN}"

_CHUNK_SIZE = 4000


def send_report(text: str) -> None:
    chunks = [text[i : i + _CHUNK_SIZE] for i in range(0, len(text), _CHUNK_SIZE)]
    for index, chunk in enumerate(chunks, start=1):
        response = requests.post(
            f"{_BASE_URL}/sendMessage",
            json={
                "chat_id": _CHAT_ID,
                "text": chunk,
                "parse_mode": "Markdown",
            },
        )
        response.raise_for_status()
        print(f"[telegram] sent chunk {index}/{len(chunks)}")


def get_chat_id() -> None:
    response = requests.get(f"{_BASE_URL}/getUpdates")
    response.raise_for_status()
    updates = response.json().get("result", [])
    if not updates:
        print("[telegram] no updates found — send a message to the bot first")
        return
    last = updates[-1]
    chat = (
        last.get("message")
        or last.get("channel_post")
        or last.get("edited_message")
        or {}
    ).get("chat", {})
    print(f"[telegram] most recent chat_id: {chat.get('id')}  (type: {chat.get('type')}, title: {chat.get('title') or chat.get('username')})")
