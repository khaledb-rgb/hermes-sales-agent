"""Vercel serverless function — receives Telegram webhook POSTs and responds in ~4s."""
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler

# Project root is the parent of the api/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_agent import answer_question, generate_report
from ghl_client import fetch_all, fetch_for_qa
from github_client import save_report
from telegram_client import delete_message, edit_message, send_error, send_message, send_report

_cache: dict = {"data": None, "ts": 0.0}
_CACHE_TTL = 300  # 5 minutes

_processed_updates: set = set()  # dedup Telegram retries
_processed_lock = threading.Lock()


def _get_cached_data() -> dict | None:
    if _cache["data"] is not None and time.time() - _cache["ts"] < _CACHE_TTL:
        return _cache["data"]
    return None


def _set_cache(data: dict) -> None:
    _cache["data"] = data
    _cache["ts"] = time.time()


def _extract_question(text: str, entities: list) -> str:
    """
    Strip @mention entities from text using Telegram's entity offsets.
    Works no matter where in the message the mention appears:
    '@Bot question', 'question @Bot', 'question @Bot more'.
    Falls back to a simple regex if entities are missing.
    """
    mention_ents = sorted(
        [e for e in entities if e.get("type") == "mention"],
        key=lambda e: e["offset"],
        reverse=True,  # remove from the end so earlier offsets stay valid
    )
    for ent in mention_ents:
        start, length = ent["offset"], ent["length"]
        text = text[:start] + text[start + length:]

    if not mention_ents:
        # Fallback: strip any @word (handles case where entities weren't sent)
        text = re.sub(r"@\w+", "", text)

    return text.strip()


def _handle_update(update: dict) -> None:
    message = update.get("message", {})
    text = message.get("text", "").strip()
    chat_id = str(message.get("chat", {}).get("id", ""))
    entities = message.get("entities", [])

    if not text or not chat_id:
        return

    # Detect entity types
    has_mention = any(e.get("type") == "mention" for e in entities)
    has_command = any(e.get("type") == "bot_command" for e in entities)

    # /report command — also handles /report@BotName format used in multi-bot groups
    base_cmd = text.split()[0].split("@")[0].lower()  # "/report@bot" → "/report"
    if base_cmd == "/report":
        send_message(chat_id, "_Generating daily report..._")
        data = fetch_all()
        report = generate_report(data)
        send_report(report)
        save_report(report)
        return

    # Ignore all other slash commands
    if has_command:
        return

    # Only respond when the bot is explicitly mentioned
    if not has_mention and "@" not in text:
        return

    # Extract the actual question (strip @mentions wherever they appear)
    question = _extract_question(text, entities)

    if not question:
        return

    # Use cached GHL data if fresh; otherwise fetch in parallel with thinking message
    cached = _get_cached_data()
    fetch_thread = None
    ghl_result: list = [None]
    ghl_error: list = [None]

    if cached is None:
        def _fetch():
            try:
                ghl_result[0] = fetch_for_qa()
            except Exception as exc:
                ghl_error[0] = exc

        fetch_thread = threading.Thread(target=_fetch, daemon=True)
        fetch_thread.start()

    thinking_msg_id = send_message(chat_id, "_Hermes is thinking..._")

    if fetch_thread is not None:
        fetch_thread.join(timeout=8)
        if ghl_error[0]:
            if thinking_msg_id:
                delete_message(chat_id, thinking_msg_id)
            send_error(str(ghl_error[0]))
            return
        cached = ghl_result[0] or {}
        _set_cache(cached)

    try:
        answer = answer_question(question, cached)
        if thinking_msg_id:
            edit_message(chat_id, thinking_msg_id, answer)  # replace thinking msg in-place
        else:
            send_message(chat_id, answer)
    except Exception as e:
        if thinking_msg_id:
            delete_message(chat_id, thinking_msg_id)
        send_error(str(e))


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Keep-warm — also pre-fills GHL cache so Q&A requests are fast."""
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")
        # Refresh cache in background so the next Q&A skips the GHL fetch
        def _warm():
            try:
                _set_cache(fetch_for_qa())
            except Exception as e:
                print(f"[webhook] cache warm error: {e}")
        threading.Thread(target=_warm, daemon=True).start()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        # Respond immediately — Vercel buffers until do_POST returns, so we
        # spawn a background thread and return right away.
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok": true}')

        try:
            update = json.loads(body)
            update_id = update.get("update_id")
            with _processed_lock:
                if update_id in _processed_updates:
                    return  # duplicate retry — drop silently
                _processed_updates.add(update_id)
                if len(_processed_updates) > 500:
                    _processed_updates.clear()
            _handle_update(update)  # synchronous — keeps Vercel function alive
        except Exception as e:
            print(f"[webhook] error: {e}")

    def log_message(self, *args):
        pass  # suppress built-in request logging
