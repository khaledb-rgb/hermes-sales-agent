import json
import os
import re
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_system_prompt_path = Path(__file__).parent / "prompts" / "system_prompt.txt"

_QA_SYSTEM = """You are Hermes, an AI assistant embedded in a sales team's Telegram group.
You have access to live CRM data from GoHighLevel: contacts, opportunities, users, and pipelines.

The data has two parts:
- "summary": pre-computed aggregates covering ALL records (use these for totals and counts).
- "contacts" / "opportunities": the most recent individual records (up to 300 contacts, 200 opportunities).
  If you need to list specific names, use these. The summary totals are authoritative for overall counts.

OUTPUT MEDIUM: Your replies are sent via the Telegram Bot API with parse_mode=Markdown. \
Telegram renders triple-backtick fenced blocks as monospace code — ASCII tables display perfectly inside them. \
You have NO technical limitation on producing tables. Always output them directly when asked.

Rules:
- Answer only from the data provided. Never invent or estimate figures.
- For total counts and breakdowns, always use the "summary" section — it covers all records.
- NEVER truncate lists with "and X more" or "..." — always show every item in full.
- Give detailed, structured answers. Break down numbers, list names, show totals.
- If asked about leads: give the count (from summary), list names, sources, and who they're assigned to.
- If asked about deals: give count, total pipeline value, stage breakdown, and assigned reps.
- If asked about a rep: show all their deals, contacts, values, and stages.
- If asked about pipeline: show each stage with count and total value.
- Always end with a short *Summary* line with the key takeaway.
- Use Telegram Markdown: *bold* for names/numbers/totals, `code` for stages/IDs, _italic_ for labels.
- If the data does not contain enough information to answer, say so clearly."""

_TABLE_INSTRUCTION = """

Respond with a pipe-and-dash ASCII table inside a triple-backtick code fence. \
Rules for a clean table:
- Header row, then separator row of dashes, then data rows.
- Max 4 columns — pick only the most relevant fields.
- Truncate cell values to 14 chars max (add … if cut).
- Dates are already formatted as YYYY-MM-DD — keep them as-is.
- Rep names are already resolved — never show IDs.
- Keep each row on a single line.
Example:
```
| Name           | Source   | Rep          | Date       |
|----------------|----------|--------------|------------|
| John Smith     | Calendly | Egor K.      | 2026-06-12 |
```"""


def _pick(record: dict, keys: list) -> dict:
    return {k: v for k, v in record.items() if k in keys and v not in (None, "", [], {})}


def _fmt_date(val: str) -> str:
    """2026-06-12T16:14:28.130Z → 2026-06-12"""
    return val[:10] if val and "T" in val else (val or "")


def _slim_for_qa(data: dict) -> dict:
    """
    Compact context for Q&A.

    Sends summary stats (covering ALL records) plus the most recent
    300 contacts and 200 opportunities as individual records.
    This keeps the payload under ~30k tokens regardless of CRM size.
    """
    user_map = {
        u["id"]: f"{u.get('firstName', '')} {u.get('lastName', '')}".strip()
        for u in data.get("users", [])
        if u.get("id")
    }

    def rep(uid: str) -> str:
        return user_map.get(uid, uid or "")

    # Build full contact list (slim fields) for summary stats
    all_contacts = [
        {
            "name": f"{c.get('firstName', '')} {c.get('lastName', '')}".strip(),
            "source": c.get("source", "") or "Unknown",
            "assignedTo": rep(c.get("assignedTo", "")),
            "dateAdded": _fmt_date(c.get("dateAdded", "")),
            "tags": c.get("tags", []),
        }
        for c in data.get("contacts", [])
    ]

    all_opps = [
        {
            "name": o.get("name", ""),
            "monetaryValue": o.get("monetaryValue", 0) or 0,
            "status": o.get("status", ""),
            "stage": o.get("pipelineStageName", ""),
            "assignedTo": rep(o.get("assignedTo", "")),
            "updatedAt": _fmt_date(o.get("updatedAt", "")),
            "expectedCloseDate": _fmt_date(o.get("expectedCloseDate", "")),
        }
        for o in data.get("opportunities", [])
    ]

    # Summary stats — computed over ALL records
    contact_by_source = dict(Counter(c["source"] for c in all_contacts).most_common())
    contact_by_rep = dict(Counter(c["assignedTo"] for c in all_contacts).most_common())

    open_opps = [o for o in all_opps if o["status"] == "open"]
    opp_by_stage = dict(Counter(o["stage"] for o in open_opps).most_common())
    opp_by_rep = dict(Counter(o["assignedTo"] for o in open_opps).most_common())
    opp_value_by_rep = {}
    for o in open_opps:
        opp_value_by_rep[o["assignedTo"]] = (
            opp_value_by_rep.get(o["assignedTo"], 0) + o["monetaryValue"]
        )

    summary = {
        "contacts": {
            "total": len(all_contacts),
            "by_source": contact_by_source,
            "by_rep": contact_by_rep,
        },
        "opportunities": {
            "total": len(all_opps),
            "open": len(open_opps),
            "open_value": sum(o["monetaryValue"] for o in open_opps),
            "by_stage": opp_by_stage,
            "by_rep": opp_by_rep,
            "value_by_rep": opp_value_by_rep,
        },
    }

    # Most recent individual records (sorted by date descending)
    recent_contacts = sorted(
        all_contacts, key=lambda c: c["dateAdded"], reverse=True
    )[:300]

    recent_opps = sorted(
        all_opps, key=lambda o: o["updatedAt"], reverse=True
    )[:200]

    return {
        "summary": summary,
        "contacts": recent_contacts,
        "opportunities": recent_opps,
        "users": [
            {"name": f"{u.get('firstName', '')} {u.get('lastName', '')}".strip()}
            for u in data.get("users", [])
        ],
        "pipelines": [
            {"name": p.get("name"), "stages": [s.get("name") for s in p.get("stages", [])]}
            for p in data.get("pipelines", [])
        ],
    }


def _slim_for_report(data: dict) -> dict:
    """Fuller context for the daily report."""
    return {
        "contacts": [
            _pick(c, ["id", "firstName", "lastName", "email", "phone", "type",
                      "source", "assignedTo", "dateAdded", "tags", "leadSource"])
            for c in data.get("contacts", [])
        ],
        "opportunities": [
            _pick(o, ["id", "name", "monetaryValue", "status", "pipelineStageName",
                      "assignedTo", "updatedAt", "expectedCloseDate", "contactId"])
            for o in data.get("opportunities", [])
        ],
        "users": [
            _pick(u, ["id", "firstName", "lastName", "email"])
            for u in data.get("users", [])
        ],
        "pipelines": [
            {"name": p.get("name"),
             "stages": [s.get("name") for s in p.get("stages", [])]}
            for p in data.get("pipelines", [])
        ],
        "invoices": [
            _pick(i, ["id", "status", "total", "dueDate"])
            for i in data.get("invoices", [])
        ],
    }


def generate_report(data: dict) -> str:
    system = _system_prompt_path.read_text(encoding="utf-8")
    user_message = (
        "Here is today's CRM data. Generate the sales report:\n\n"
        + json.dumps(_slim_for_report(data), ensure_ascii=False, default=str)
    )
    try:
        response = _client.chat.completions.create(
            model="gpt-4o",
            max_tokens=2048,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content or ""
    except OpenAIError as e:
        return f"[Hermes] API error: {e}"


_TABLE_KEYWORDS = {"table", "tableau", "جدول", "tabela", "tabell", "teble"}

_TS_RE = re.compile(r'(\d{4}-\d{2}-\d{2})T\d{2}:\d{2}:\d{2}[\.\d]*Z?')


def _clean_output(text: str) -> str:
    """Strip time component from any ISO timestamp and wrap bare pipe tables in code fences."""
    text = _TS_RE.sub(r'\1', text)

    lines = text.split("\n")
    result, buf, in_code = [], [], False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            if buf:
                result += ["```"] + buf + ["```"]
                buf = []
            result.append(line)
        elif not in_code and stripped.startswith("|"):
            buf.append(line)
        else:
            if buf:
                result += ["```"] + buf + ["```"]
                buf = []
            result.append(line)
    if buf:
        result += ["```"] + buf + ["```"]
    return "\n".join(result)


def answer_question(question: str, data: dict) -> str:
    wants_table = any(kw in question.lower() for kw in _TABLE_KEYWORDS)
    suffix = _TABLE_INSTRUCTION if wants_table else ""
    user_message = (
        "Here is the current CRM data:\n\n"
        + json.dumps(_slim_for_qa(data), ensure_ascii=False, default=str)
        + f"\n\nQuestion: {question}{suffix}"
    )
    try:
        response = _client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=4000,
            messages=[
                {"role": "system", "content": _QA_SYSTEM},
                {"role": "user", "content": user_message},
            ],
        )
        return _clean_output(response.choices[0].message.content or "")
    except OpenAIError as e:
        raise RuntimeError(f"OpenAI error: {e}") from e
