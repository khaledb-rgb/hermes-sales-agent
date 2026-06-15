"""One-shot daily report — fetch GHL data, generate report, send to group, save to GitHub."""
import os
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

# The workflow triggers at 13:00 and 14:00 UTC to bracket US Central DST. On a
# scheduled run, proceed only at 08:00 America/Chicago so the report fires exactly
# once per day. Manual (workflow_dispatch) runs always proceed.
if os.getenv("GITHUB_EVENT_NAME") == "schedule":
    try:
        from zoneinfo import ZoneInfo

        hour = datetime.now(ZoneInfo("America/Chicago")).hour
    except Exception:
        hour = datetime.utcnow().hour
    if hour != 8:
        print(f"[report] {hour:02d}:00 America/Chicago is not 08:00 — skipping this trigger")
        raise SystemExit(0)

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
