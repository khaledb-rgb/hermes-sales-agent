import os
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
    try:
        r = requests.get(
            f"{BASE_URL}{path}",
            headers=HEADERS,
            params=params or {},
            timeout=_REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print(f"[ghl] GET {path} error: {e}")
        return {}


def get_contacts(max_pages: int = 0) -> list:
    """Fetch contacts using cursor-based pagination.

    max_pages=0 means fetch all pages. max_pages=N stops after N pages.
    """
    results, after, page_count = [], None, 0
    while True:
        params = {"locationId": LOCATION_ID, "limit": 100}
        if after:
            params["startAfterId"] = after
        data = _get("/contacts/", params)
        page = data.get("contacts", [])
        results.extend(page)
        page_count += 1
        meta = data.get("meta", {})
        after = meta.get("startAfterId") or meta.get("nextPageUrl")
        if not after or len(page) < 100:
            break
        if max_pages and page_count >= max_pages:
            break
    return results


def get_opportunities(max_pages: int = 0) -> list:
    """Fetch opportunities using cursor-based pagination.

    max_pages=0 means fetch all pages. max_pages=N stops after N pages.
    """
    results, after, page_count = [], None, 0
    while True:
        params = {"location_id": LOCATION_ID, "limit": 100}
        if after:
            params["startAfterId"] = after
        data = _get("/opportunities/search", params)
        page = data.get("opportunities", [])
        results.extend(page)
        page_count += 1
        meta = data.get("meta", {})
        after = meta.get("startAfterId") or meta.get("nextPageUrl")
        if not after or len(page) < 100:
            break
        if max_pages and page_count >= max_pages:
            break
    return results


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


def fetch_for_qa() -> dict:
    """Fetch Q&A-relevant GHL data in parallel.

    Contacts and opportunities are capped at 3 pages (300 records each)
    so a cold-start response fits well within Telegram's 60s read timeout.
    The claude_agent layer computes summary stats from these records and
    sends only the most recent 300/200 to OpenAI anyway.
    """
    fetchers = {
        "contacts": lambda: get_contacts(max_pages=3),
        "opportunities": lambda: get_opportunities(max_pages=2),
        "users": get_users,
        "pipelines": get_pipelines,
    }
    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
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
