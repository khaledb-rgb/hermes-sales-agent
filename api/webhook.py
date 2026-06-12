"""Vercel serverless function — receives Telegram webhook POSTs and responds in ~4s."""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler

# Project root is the parent of the api/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_agent import answer_question, generate_report
from ghl_client import fetch_all, fetch_for_qa
from github_client import save_report
from telegram_client import send_error, send_message, send_report


def _handle_update(update: dict) -> None:
    message = update.get("message", {})
    text = message.get("text", "").strip()
    chat_id = str(message.get("chat", {}).get("id", ""))

    if not text or not chat_id:
        return

    # Ignore bot commands except /report
    if text.startswith("/") and not text.lower().startswith("/report"):
        return

    # Strip @BotMention prefix (group messages tagged at the bot)
    if text.startswith("@"):
        text = text.split(" ", 1)[1].strip() if " " in text else ""
    if not text:
        return

    if text.lower() == "/report":
        send_message(chat_id, "_Generating daily report..._")
        data = fetch_all()
        report = generate_report(data)
        send_report(report)
        save_report(report)
        return

    # Q&A: kick off GHL fetch and "thinking" message in parallel, then call OpenAI
    ghl_result: list = [None]
    ghl_error: list = [None]

    def _fetch():
        try:
            ghl_result[0] = fetch_for_qa()
        except Exception as exc:
            ghl_error[0] = exc

    fetch_thread = threading.Thread(target=_fetch, daemon=True)
    fetch_thread.start()

    send_message(chat_id, "_Hermes is thinking..._")  # runs while GHL fetches

    fetch_thread.join(timeout=8)
    if ghl_error[0]:
        send_error(str(ghl_error[0]))
        return

    try:
        answer = answer_question(text, ghl_result[0] or {})
        send_message(chat_id, answer)
    except Exception as e:
        send_error(str(e))


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Keep-warm health check — triggered every 5 min by GitHub Actions cron."""
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            update = json.loads(body)
            _handle_update(update)
        except Exception as e:
            print(f"[webhook] error: {e}")
        finally:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok": true}')

    def log_message(self, *args):
        pass  # suppress built-in request logging
