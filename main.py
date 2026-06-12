from dotenv import load_dotenv

load_dotenv()

from claude_agent import generate_report
from ghl_client import fetch_all
from github_client import save_report
from telegram_client import send_report


def run() -> None:
    print("Fetching GHL data...")
    data = fetch_all()
    print(
        f"Fetched: {len(data.get('contacts', []))} contacts, "
        f"{len(data.get('opportunities', []))} opportunities, "
        f"{len(data.get('appointments', []))} appointments"
    )

    print("Generating report with Hermes...")
    report = generate_report(data)
    print(f"Preview:\n{report[:300]}\n...")

    print("Sending to Telegram...")
    send_report(report)

    print("Saving to GitHub...")
    save_report(report)

    print("Done. Report delivered.")


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"[hermes] failed: {e}")
