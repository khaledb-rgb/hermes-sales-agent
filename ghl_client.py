import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://services.leadconnectorhq.com"
API_KEY = os.getenv("GHL_API_KEY")
LOCATION_ID = os.getenv("GHL_LOCATION_ID")
COMPANY_ID = os.getenv("GHL_COMPANY_ID")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Version": "2021-07-28",
    "Content-Type": "application/json",
}

_REQUEST_TIMEOUT = 8  # seconds per individual GHL API call


def _get(path: str, params: dict = None) -> dict:
    for attempt in range(4):
        try:
            r = requests.get(
                f"{BASE_URL}{path}",
                headers=HEADERS,
                params=params or {},
                timeout=_REQUEST_TIMEOUT,
            )
            if r.status_code == 429 and attempt < 3:
                wait = float(r.headers.get("Retry-After", 2 ** attempt))
                print(f"[ghl] 429 on {path} — retrying in {wait:.0f}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            print(f"[ghl] GET {path} error: {e}")
            return {}
    return {}


def get_contacts(max_pages: int = 0) -> list:
    """Fetch contacts using cursor-based pagination.

    max_pages=0 means fetch all pages. max_pages=N stops after N pages.
    """
    results, page_count = [], 0
    start_after = start_after_id = None
    while True:
        params = {"locationId": LOCATION_ID, "limit": 100}
        if start_after_id:
            # GHL's cursor needs BOTH startAfter (ms timestamp) and startAfterId;
            # sending only one returns the same first page forever.
            params["startAfter"] = start_after
            params["startAfterId"] = start_after_id
        data = _get("/contacts/", params)
        page = data.get("contacts", [])
        results.extend(page)
        page_count += 1
        meta = data.get("meta", {})
        start_after = meta.get("startAfter")
        start_after_id = meta.get("startAfterId")
        if not start_after_id or len(page) < 100:
            break
        if max_pages and page_count >= max_pages:
            break
        if page_count >= 500:  # safety cap (~50k records) against runaway loops
            break
    return results


def get_opportunities(max_pages: int = 0) -> list:
    """Fetch opportunities using cursor-based pagination.

    max_pages=0 means fetch all pages. max_pages=N stops after N pages.
    """
    results, page_count = [], 0
    start_after = start_after_id = None
    while True:
        params = {"location_id": LOCATION_ID, "limit": 100}
        if start_after_id:
            # GHL's cursor needs BOTH startAfter (ms timestamp) and startAfterId;
            # sending only one returns the same first page forever.
            params["startAfter"] = start_after
            params["startAfterId"] = start_after_id
        data = _get("/opportunities/search", params)
        page = data.get("opportunities", [])
        results.extend(page)
        page_count += 1
        meta = data.get("meta", {})
        start_after = meta.get("startAfter")
        start_after_id = meta.get("startAfterId")
        if not start_after_id or len(page) < 100:
            break
        if max_pages and page_count >= max_pages:
            break
        if page_count >= 500:  # safety cap (~50k records) against runaway loops
            break
    return results


def get_total(resource: str) -> int:
    """Exact record count for a resource, read from the search endpoint's meta.

    Costs a single page fetch (limit=1) rather than paginating the whole CRM, so
    the bot can report true totals even though it only loads the most-recent N
    records for listing. Returns 0 if the count is unavailable.
    """
    if resource == "opportunities":
        data = _get("/opportunities/search", {"location_id": LOCATION_ID, "limit": 1})
    elif resource == "contacts":
        data = _get("/contacts/", {"locationId": LOCATION_ID, "limit": 1})
    else:
        return 0
    return data.get("meta", {}).get("total", 0) or 0


def get_users() -> list:
    return _get("/users/", {"locationId": LOCATION_ID}).get("users", [])


def get_pipelines() -> list:
    return _get("/opportunities/pipelines", {"locationId": LOCATION_ID}).get("pipelines", [])


def get_conversations() -> list:
    return _get("/conversations/search", {"locationId": LOCATION_ID, "limit": 50}).get("conversations", [])


def get_invoices() -> list:
    return _get("/invoices/", {
        "altId": LOCATION_ID,
        "altType": "location",
        "offset": "0",
        "limit": "50",
    }).get("invoices", [])


def get_contact_appointments(contact_id: str) -> list:
    return _get(f"/contacts/{contact_id}/appointments").get("events", [])


def get_appointments(days_back: int = 1, days_ahead: int = 21) -> list:
    """Calendar events across all location users in a time window.

    GHL's /calendars/events requires a userId/calendarId. There are ~17 users
    vs. ~120 calendars here, so we fan out by user (in parallel) and dedup by
    event id. Window defaults to yesterday..3 weeks ahead — enough for "who's
    booked today/this week" without pulling appointment history.
    """
    now = int(time.time() * 1000)
    start = now - days_back * 86_400_000
    end = now + days_ahead * 86_400_000
    users = get_users()

    def fetch(uid: str) -> list:
        return _get(
            "/calendars/events",
            {"locationId": LOCATION_ID, "userId": uid, "startTime": start, "endTime": end},
        ).get("events", [])

    seen, events = set(), []
    with ThreadPoolExecutor(max_workers=6) as executor:
        for evs in executor.map(lambda u: fetch(u.get("id", "")), users):
            for e in evs:
                eid = e.get("id")
                if eid and eid not in seen:
                    seen.add(eid)
                    events.append(e)
    return events


def fetch_for_qa() -> dict:
    """Fetch Q&A-relevant GHL data in parallel.

    Contacts and opportunities are capped at 5 pages (500 records each) — the
    most-recent records, used for listing and breakdowns. True totals come
    separately and cheaply from get_total() (the search meta), so the bot can
    report exact whole-CRM counts without paginating ~163k records live.
    """
    fetchers = {
        "contacts": lambda: get_contacts(max_pages=5),
        "opportunities": lambda: get_opportunities(max_pages=5),
        "users": get_users,
        "pipelines": get_pipelines,
        "contacts_total": lambda: get_total("contacts"),
        "opportunities_total": lambda: get_total("opportunities"),
        "conversations": get_conversations,
        "invoices": get_invoices,
        "appointments": get_appointments,
    }
    results = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fn): key for key, fn in fetchers.items()}
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as e:
                print(f"[ghl] {key} fetch error: {e}")
                results[key] = []
    return results


def fetch_all() -> dict:
    """Fetch all GHL data in parallel (used for the daily report)."""
    fetchers = {
        "contacts": get_contacts,
        "opportunities": get_opportunities,
        "users": get_users,
        "pipelines": get_pipelines,
        "conversations": get_conversations,
        "invoices": get_invoices,
    }
    results = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(fn): key for key, fn in fetchers.items()}
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as e:
                print(f"[ghl] {key} fetch error: {e}")
                results[key] = []
    return results
