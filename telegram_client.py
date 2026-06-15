import os
import re
import time
from typing import Callable

import requests
from dotenv import load_dotenv

load_dotenv()

# Reports may bundle several Telegram messages into one string, separated by a
# line containing only "---SPLIT---". send_report() sends each as its own message.
SPLIT_MARKER = "---SPLIT---"
_SPLIT_RE = re.compile(r"(?m)^[ \t]*-{3}\s*SPLIT\s*-{3}[ \t]*$")

_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
_ADMIN_ID = os.getenv("TELEGRAM_ADMIN_ID")
_BASE_URL = f"https://api.telegram.org/bot{_TOKEN}"

_CHUNK_SIZE = 4000


def send_typing(chat_id: str) -> None:
    """Show the built-in Telegram typing indicator (lasts ~5s, disappears on next message)."""
    try:
        requests.post(
            f"{_BASE_URL}/sendChatAction",
            json={"chat_id": chat_id, "action": "typing"},
            timeout=5,
        )
    except Exception:
        pass


def _post_chunk(chat_id: str, chunk: str) -> int | None:
    """Send one chunk as Markdown; if Telegram can't parse it, resend as plain text.

    Real CRM data (names/sources with _, *, [, `) can break Telegram's legacy
    Markdown parser and return HTTP 400. Falling back to plain text guarantees the
    message is always delivered rather than lost.
    """
    for parse_mode in ("Markdown", None):
        payload = {"chat_id": chat_id, "text": chunk}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        response = requests.post(f"{_BASE_URL}/sendMessage", json=payload)
        if response.ok:
            return response.json().get("result", {}).get("message_id")
        if parse_mode:  # Markdown rejected — log and retry as plain text
            try:
                desc = response.json().get("description")
            except Exception:
                desc = response.text[:200]
            print(f"[telegram] Markdown send failed ({desc}) — resending as plain text")
            continue
        response.raise_for_status()  # plain text also failed: a real error
    return None


def send_message(chat_id: str, text: str) -> int | None:
    """Send a message and return the message_id of the last sent chunk."""
    chunks = [text[i : i + _CHUNK_SIZE] for i in range(0, len(text), _CHUNK_SIZE)]
    message_id = None
    for index, chunk in enumerate(chunks, start=1):
        message_id = _post_chunk(chat_id, chunk)
        if len(chunks) > 1:
            print(f"[telegram] sent chunk {index}/{len(chunks)}")
    return message_id


def edit_message(chat_id: str, message_id: int, text: str) -> bool:
    """Edit an existing message in-place. Returns True on success."""
    chunks = [text[i : i + _CHUNK_SIZE] for i in range(0, len(text), _CHUNK_SIZE)]
    ok = False
    for parse_mode in ("Markdown", None):  # try Markdown first, fall back to plain
        try:
            payload = {"chat_id": chat_id, "message_id": message_id, "text": chunks[0]}
            if parse_mode:
                payload["parse_mode"] = parse_mode
            resp = requests.post(f"{_BASE_URL}/editMessageText", json=payload, timeout=10)
            ok = resp.json().get("ok", False)
            if ok:
                break
            print(f"[telegram] editMessageText failed ({parse_mode}): {resp.json().get('description')}")
        except Exception as e:
            print(f"[telegram] editMessageText error: {e}")
    if ok:
        for chunk in chunks[1:]:
            send_message(chat_id, chunk)
    return ok


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


def _split_parts(text: str) -> list:
    """Split a report into separate messages on the ---SPLIT--- marker."""
    return [part.strip() for part in _SPLIT_RE.split(text) if part.strip()]


def send_report(text: str) -> None:
    """Send a report to the group, one Telegram message per ---SPLIT--- section."""
    for part in _split_parts(text):
        send_message(_CHAT_ID, part)


def archive_text(text: str) -> str:
    """Collapse the multi-message split marker into one clean document for saving,
    and drop the Telegram-only underscore escaping so the saved file reads cleanly."""
    return _SPLIT_RE.sub("———", text).strip().replace("\\_", "_")


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
