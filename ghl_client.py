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


def _get(path: str, params: dict = None) -> dict:
    try:
        r = requests.get(f"{BASE_URL}{path}", headers=HEADERS, params=params or {})
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print(f"[ghl] GET {path} error: {e}")
        return {}


def get_contacts() -> list:
    return _get("/contacts/", {"locationId": LOCATION_ID, "limit": 100}).get("contacts", [])


def get_opportunities() -> list:
    return _get("/opportunities/search", {"location_id": LOCATION_ID, "limit": 100}).get("opportunities", [])


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


def fetch_all() -> dict:
    """Fetch all GHL data in parallel — ~1.5s instead of ~6s sequential."""
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
