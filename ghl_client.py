import os
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


def get_contacts() -> list:
    try:
        response = requests.get(
            f"{BASE_URL}/contacts/",
            headers=HEADERS,
            params={"locationId": LOCATION_ID, "limit": 100},
        )
        response.raise_for_status()
        return response.json().get("contacts", [])
    except requests.RequestException as e:
        print(f"[ghl] get_contacts error: {e}")
        return []


def get_opportunities() -> list:
    try:
        response = requests.get(
            f"{BASE_URL}/opportunities/search",
            headers=HEADERS,
            params={"location_id": LOCATION_ID, "limit": 100},
        )
        response.raise_for_status()
        return response.json().get("opportunities", [])
    except requests.RequestException as e:
        print(f"[ghl] get_opportunities error: {e}")
        return []


def get_appointments() -> list:
    try:
        response = requests.get(
            f"{BASE_URL}/calendars/appointments",
            headers=HEADERS,
            params={"locationId": LOCATION_ID, "limit": 100},
        )
        response.raise_for_status()
        return response.json().get("appointments", [])
    except requests.RequestException as e:
        print(f"[ghl] get_appointments error: {e}")
        return []


def get_users() -> list:
    try:
        response = requests.get(
            f"{BASE_URL}/users/",
            headers=HEADERS,
            params={"companyId": COMPANY_ID},
        )
        response.raise_for_status()
        return response.json().get("users", [])
    except requests.RequestException as e:
        print(f"[ghl] get_users error: {e}")
        return []


def fetch_all() -> dict:
    return {
        "contacts": get_contacts(),
        "opportunities": get_opportunities(),
        "appointments": get_appointments(),
        "users": get_users(),
    }
