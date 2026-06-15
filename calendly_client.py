"""Calendly API client — PLACEHOLDER.

Reserved for the Calendly integration. A token will be supplied soon; set it as
CALENDLY_API_KEY in .env (and as a GitHub/Vercel secret) to activate.

Once wired up, this feeds the daily report so the KPI row can show real
*Bookings* and *Show rate* figures (see prompts/system_prompt.txt title
"Daily CRM + Calendly Report"). Until then every function safely returns empty
data and the report falls back to CRM-only KPIs — nothing breaks.
"""
import os

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("CALENDLY_API_KEY")
BASE_URL = "https://api.calendly.com"


def is_configured() -> bool:
    """True once a Calendly token is present."""
    return bool(API_KEY)


def fetch_bookings() -> list:
    """Return Calendly scheduled events for the report.

    Returns [] until the integration is implemented, so callers can merge the
    result unconditionally without breaking.

    TODO (when CALENDLY_API_KEY arrives):
      - GET /scheduled_events with the org/user URI + a date range
      - page through `collection` via `pagination.next_page`
      - normalize each event to:
          {"name", "invitee", "status", "start_time", "event_type"}
      so claude_agent can compute Bookings (count) and Show rate
      (attended / total) for the KPI row.
    """
    if not API_KEY:
        return []
    print("[calendly] CALENDLY_API_KEY is set but the client is not implemented yet.")
    return []
