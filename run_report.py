"""One-shot daily report — fetch GHL data, generate report, send to group, save to GitHub."""
import os

from dotenv import load_dotenv

load_dotenv()

from claude_agent import generate_report
from ghl_client import fetch_all
from github_client import save_report
from telegram_client import archive_text, send_message, send_report

chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
send_message(chat_id, "_Generating daily report..._")
data = fetch_all()
report = generate_report(data)
send_report(report)
save_report(archive_text(report))
print("[report] Done.")
