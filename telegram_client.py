import os
import time
from typing import Callable

import requests
from dotenv import load_dotenv

load_dotenv()

_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
_ADMIN_ID = os.getenv("TELEGRAM_ADMIN_ID")
_BASE_URL = f"https://api.telegram.org/bot{_TOKEN}"

_CHUNK_SIZE = 4000


def send_message(chat_id: str, text: str) -> int | None:
    """Send a message and return the message_id of the last sent chunk."""
    chunks = [text[i : i + _CHUNK_SIZE] for i in range(0, len(text), _CHUNK_SIZE)]
    message_id = None
    for index, chunk in enumerate(chunks, start=1):
        response = requests.post(
            f"{_BASE_URL}/sendMessage",
            json={"chat_id": chat_id, "text": chunk, "parse_mode": "Markdown"},
        )
        response.raise_for_status()
        message_id = response.json().get("result", {}).get("message_id")
        if len(chunks) > 1:
            print(f"[telegram] sent chunk {index}/{len(chunks)}")
    return message_id


def edit_message(chat_id: str, message_id: int, text: str) -> None:
    """Edit an existing message in-place (replaces thinking msg with the answer)."""
    chunks = [text[i : i + _CHUNK_SIZE] for i in range(0, len(text), _CHUNK_SIZE)]
    try:
        requests.post(
            f"{_BASE_URL}/editMessageText",
            json={"chat_id": chat_id, "message_id": message_id,
                  "text": chunks[0], "parse_mode": "Markdown"},
        )
    except Exception as e:
        print(f"[telegram] could not edit message {message_id}: {e}")
    # If answer is longer than one chunk, send the rest as new messages
    for chunk in chunks[1:]:
        send_message(chat_id, chunk)


def delete_message(chat_id: str, message_id: int) -> None:
    try:
        requests.post(
            f"{_BASE_URL}/deleteMessage",
            json={"chat_id": chat_id, "message_id": message_id},
        )
    except Exception as e:
        print(f"[telegram] could not delete message {message_id}: {e}")


def send_error(error: str) -> None:
    """Send full error details privately to admin only."""
    if _ADMIN_ID:
        try:
            send_message(_ADMIN_ID, f"⚠️ *Hermes error:*\n`{error}`")
        except Exception as e:
            print(f"[telegram] could not send error to admin: {e}")
    print(f"[telegram] error: {error}")


def send_report(text: str) -> None:
    send_message(_CHAT_ID, text)


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


def start_polling(handler: Callable[[str, str], None]) -> None:
    offset = 0
    print("[telegram] bot started, listening for messages...")

    while True:
        try:
            response = requests.get(
                f"{_BASE_URL}/getUpdates",
                params={"offset": offset, "timeout": 30, "allowed_updates": ["message"]},
                timeout=35,
            )
            response.raise_for_status()
            updates = response.json().get("result", [])

            for update in updates:
                offset = update["update_id"] + 1
                message = update.get("message", {})
                text = message.get("text", "").strip()
                chat_id = str(message.get("chat", {}).get("id", ""))

                if not text or not chat_id:
                    continue

                if text.startswith("/") and not text.startswith("/report"):
                    continue

                print(f"[telegram] message from {chat_id}: {text[:80]}")
                try:
                    handler(text, chat_id)
                except Exception as e:
                    send_error(str(e))

        except requests.RequestException as e:
            print(f"[telegram] polling error: {e} — retrying in 5s")
            time.sleep(5)
