import time

from dotenv import load_dotenv

load_dotenv()

from claude_agent import answer_question, generate_report
from ghl_client import fetch_all
from github_client import save_report
from telegram_client import send_message, send_report, start_polling

# GHL data cache — refreshed every 5 minutes
_cache: dict = {"data": None, "ts": 0.0}
_CACHE_TTL = 300


def _get_data() -> dict:
    if time.time() - _cache["ts"] > _CACHE_TTL:
        print("[main] refreshing GHL data...")
        _cache["data"] = fetch_all()
        _cache["ts"] = time.time()
        counts = {k: len(v) for k, v in _cache["data"].items()}
        print(f"[main] data loaded: {counts}")
    return _cache["data"]


def handle_message(text: str, chat_id: str) -> None:
    # strip bot mention if present (e.g. "@HermesBot what are today's leads?")
    if text.startswith("@"):
        text = text.split(" ", 1)[1].strip() if " " in text else ""
    if not text:
        return

    if text.lower() == "/report":
        send_message(chat_id, "_Generating daily report..._")
        data = fetch_all()
        _cache["data"] = data
        _cache["ts"] = time.time()
        report = generate_report(data)
        send_report(report)
        save_report(report)
        return

    send_message(chat_id, "_Hermes is thinking..._")
    data = _get_data()
    answer = answer_question(text, data)
    send_message(chat_id, answer)


if __name__ == "__main__":
    print("[main] Hermes starting up...")
    _get_data()
    start_polling(handle_message)
