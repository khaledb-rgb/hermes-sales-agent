import base64
import os
from datetime import date

import requests
from dotenv import load_dotenv

load_dotenv()

_TOKEN = os.getenv("GITHUB_TOKEN")
_OWNER = os.getenv("GITHUB_OWNER")
_REPO = os.getenv("GITHUB_REPO")
_BASE_URL = "https://api.github.com"

_HEADERS = {
    "Authorization": f"Bearer {_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


def save_report(report_text: str) -> None:
    today = date.today().strftime("%Y-%m-%d")
    path = f"reports/{today}.md"
    url = f"{_BASE_URL}/repos/{_OWNER}/{_REPO}/contents/{path}"
    content = base64.b64encode(report_text.encode("utf-8")).decode("utf-8")

    # check if file already exists to get its sha (required for updates)
    sha = None
    try:
        get_response = requests.get(url, headers=_HEADERS)
        if get_response.status_code == 200:
            sha = get_response.json().get("sha")
    except requests.RequestException as e:
        print(f"[github] could not check existing file: {e}")

    payload = {
        "message": f"report: {today}",
        "content": content,
    }
    if sha:
        payload["sha"] = sha

    try:
        put_response = requests.put(url, headers=_HEADERS, json=payload)
        put_response.raise_for_status()
        file_url = put_response.json().get("content", {}).get("html_url", "")
        print(f"[github] report saved: {file_url}")
    except requests.RequestException as e:
        status = e.response.status_code if e.response is not None else "N/A"
        detail = e.response.text[:200] if e.response is not None else str(e)
        print(f"[github] failed to save report (HTTP {status}): {detail}")
