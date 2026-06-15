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
from ghl_client import fetch_for_qa
from github_client import save_report
from telegram_client import archive_text, send_error, send_message, send_report, send_typing

_cache: dict = {"data": None, "ts": 0.0}
_CACHE_TTL = 180  # 3 minutes — matches keepwarm cron interval

_processed_updates: set = set()  # dedup Telegram retries within same instance
_processed_lock = threading.Lock()

# Per-chat conversation memory (best-effort, in-process). Lets the bot resolve
# follow-ups like "what about his email?". Lost on cold start; the keepwarm cron
# keeps instances warm. Stores only text turns — never the large CRM payload.
_history: dict = {}  # chat_id -> {"turns": [{role, content}, ...], "ts": float}
_history_lock = threading.Lock()
_HISTORY_TTL = 1800        # 30 min — forget a chat's context after inactivity
_HISTORY_MAX_TURNS = 8     # keep the last 8 messages (~4 Q&A pairs)


def _get_history(chat_id: str) -> list:
    with _history_lock:
        entry = _history.get(chat_id)
        if not entry or time.time() - entry["ts"] > _HISTORY_TTL:
            return []
        return list(entry["turns"])


def _append_history(chat_id: str, question: str, answer: str) -> None:
    with _history_lock:
        entry = _history.get(chat_id)
        fresh = entry and time.time() - entry["ts"] <= _HISTORY_TTL
        turns = (entry["turns"] if fresh else []) + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        _history[chat_id] = {"turns": turns[-_HISTORY_MAX_TURNS:], "ts": time.time()}


def _clear_history(chat_id: str) -> None:
    with _history_lock:
        _history.pop(chat_id, None)


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
    Works no matter where in the message the mention appears.
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
        text = re.sub(r"@\w+", "", text)

    return text.strip()


def _handle_update(update: dict) -> None:
    message = update.get("message", {})
    text = message.get("text", "").strip()
    chat_id = str(message.get("chat", {}).get("id", ""))
    entities = message.get("entities", [])

    if not text or not chat_id:
        return

    has_mention = any(e.get("type") == "mention" for e in entities)
    has_command = any(e.get("type") == "bot_command" for e in entities)

    base_cmd = text.split()[0].split("@")[0].lower()
    if base_cmd in ("/reset", "/clear", "/forget"):
        _clear_history(chat_id)
        send_message(chat_id, "_Conversation memory cleared._")
        return

    if base_cmd == "/report":
        send_message(chat_id, "_Generating daily report..._")
        # Use the warm cache or the bounded fetch — fetch_all() can exceed
        # Vercel's function timeout under GHL rate-limiting. The full-data report
        # is produced by the daily GitHub Action (run_report.py).
        data = _get_cached_data() or fetch_for_qa()
        report = generate_report(data, include_calendly=False)  # keep replies fast
        send_report(report)
        save_report(archive_text(report))
        return

    if has_command:
        return

    if not has_mention and "@" not in text:
        return

    question = _extract_question(text, entities)

    if not question:
        return

    send_typing(chat_id)

    cached = _get_cached_data()
    if cached is None:
        try:
            cached = fetch_for_qa()
            _set_cache(cached)
        except Exception as exc:
            send_error(str(exc))
            return

    try:
        history = _get_history(chat_id)
        answer = answer_question(question, cached, history=history)
        send_message(chat_id, answer)
        _append_history(chat_id, question, answer)
    except Exception as e:
        send_error(str(e))


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Keep-warm — warms GHL cache synchronously so Q&A requests are fast.

        Must be synchronous: daemon threads are killed the moment do_GET returns,
        so a background thread never actually fills the cache.
        """
        try:
            _set_cache(fetch_for_qa())
            body = b"OK - cache warmed"
        except Exception as e:
            print(f"[webhook] cache warm error: {e}")
            body = b"OK - cache warm failed"
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok": true}')

        try:
            update = json.loads(body)
            update_id = update.get("update_id")
            with _processed_lock:
                if update_id in _processed_updates:
                    return
                _processed_updates.add(update_id)
                if len(_processed_updates) > 500:
                    _processed_updates.clear()
            _handle_update(update)
        except Exception as e:
            print(f"[webhook] error: {e}")

    def log_message(self, *args):
        pass
