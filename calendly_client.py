"""Calendly API client.

Feeds the daily report's Bookings and Show-rate KPIs. Set CALENDLY_API_KEY to
enable; if it's unset or the API errors, fetch_summary() returns {} and the
report falls back to CRM-only KPIs — nothing breaks.

Definitions:
- bookings_today = active scheduled events starting today (UTC)
- show_rate      = over active events that have already ended in the last 7 days,
                   the % of invitees who showed up (no_show not set). Depends on
                   the team marking no-shows in Calendly; if they never do, it
                   reads ~100%.
"""
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("CALENDLY_API_KEY")
BASE_URL = "https://api.calendly.com"
_TIMEOUT = 10
_MAX_EVENTS_FOR_SHOW_RATE = 80  # bound the per-event invitee calls


def is_configured() -> bool:
    return bool(API_KEY)


def _get(url: str, params: dict = None) -> dict:
    if not url.startswith("http"):
        url = f"{BASE_URL}{url}"
    r = requests.get(url, headers={"Authorization": f"Bearer {API_KEY}"},
                     params=params or {}, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _org_uri() -> str | None:
    return _get("/users/me").get("resource", {}).get("current_organization")


def _list_events(org: str, min_start: str, max_start: str) -> list:
    events = []
    url, params = f"{BASE_URL}/scheduled_events", {
        "organization": org,
        "count": 100,
        "min_start_time": min_start,
        "max_start_time": max_start,
        "sort": "start_time:asc",
    }
    while True:
        data = _get(url, params)
        events.extend(data.get("collection", []))
        nxt = (data.get("pagination") or {}).get("next_page")
        if not nxt:
            return events
        url, params = nxt, None


def _invitee_counts(event_uri: str) -> tuple:
    """(total_invitees, no_shows) for one event; (0, 0) on error."""
    try:
        inv = _get(f"{event_uri}/invitees", {"count": 100}).get("collection", [])
    except requests.RequestException:
        return (0, 0)
    return (len(inv), sum(1 for i in inv if i.get("no_show")))


def fetch_summary() -> dict:
    """Return {bookings_today, show_rate, events_7d} or {} if unavailable."""
    if not API_KEY:
        return {}
    try:
        org = _org_uri()
        if not org:
            return {}

        now = datetime.now(timezone.utc)
        now_naive = now.replace(tzinfo=None)
        today = now.strftime("%Y-%m-%d")
        win_start = (now - timedelta(days=7)).strftime("%Y-%m-%dT00:00:00Z")
        win_end = (now + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")

        active = [e for e in _list_events(org, win_start, win_end)
                  if e.get("status") == "active"]
        bookings_today = sum(1 for e in active if e.get("start_time", "")[:10] == today)

        def ended(e: dict) -> bool:
            try:
                return datetime.strptime(e.get("end_time", "")[:19], "%Y-%m-%dT%H:%M:%S") < now_naive
            except ValueError:
                return False

        occurred = [e for e in active if ended(e)][:_MAX_EVENTS_FOR_SHOW_RATE]
        total_inv = no_shows = 0
        if occurred:
            with ThreadPoolExecutor(max_workers=8) as ex:
                for fut in as_completed([ex.submit(_invitee_counts, e["uri"]) for e in occurred]):
                    t, n = fut.result()
                    total_inv += t
                    no_shows += n
        show_rate = round((total_inv - no_shows) / total_inv * 100) if total_inv else None

        return {
            "bookings_today": bookings_today,
            "show_rate": show_rate,
            "events_7d": len(active),
        }
    except requests.RequestException as e:
        print(f"[calendly] error: {e}")
        return {}
