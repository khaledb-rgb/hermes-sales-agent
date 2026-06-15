import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()

# LLM provider — if OPENROUTER_API_KEY is set, route through OpenRouter (DeepSeek
# V3.1 by default); otherwise fall back to OpenAI directly. The openai SDK speaks
# to both since OpenRouter is OpenAI-compatible.
_OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
if _OPENROUTER_KEY:
    _client = OpenAI(
        api_key=_OPENROUTER_KEY,
        base_url=os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
        default_headers={
            "HTTP-Referer": "https://github.com/khaledb-rgb/hermes-sales-agent",
            "X-Title": "Hermes Sales Agent",
        },
    )
    _REPORT_MODEL = os.getenv("REPORT_MODEL", "deepseek/deepseek-chat-v3.1")
    _QA_MODEL = os.getenv("QA_MODEL", "deepseek/deepseek-chat-v3.1")
else:
    _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    _REPORT_MODEL = os.getenv("REPORT_MODEL", "gpt-4o")
    _QA_MODEL = os.getenv("QA_MODEL", "gpt-4o-mini")

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
- DEFAULT list format: plain numbered list (1. *Name* — Source — Rep), NOT a code block table.
  Code block tables are only for when the user explicitly asks for a "table".
  Plain text lists scroll naturally in Telegram; code blocks have a fixed scroll area.
- If asked about leads: give the count (from summary), then list each contact as a numbered line.
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


def _today_iso() -> str:
    return _now().strftime("%Y-%m-%d")


def _days_since(date_str: str, today: str) -> int | None:
    """Whole days between an ISO date and today; None if unparseable."""
    if not date_str or not today:
        return None
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d")
        t = datetime.strptime(today, "%Y-%m-%d")
        return (t - d).days
    except ValueError:
        return None


def _slim_for_report(data: dict) -> dict:
    """Compact, pre-computed report context.

    Sending every contact/opportunity overflowed the model context (the old
    version requested tens of millions of tokens and always failed). Instead we
    compute the KPIs here over ALL records and pass a small summary plus only
    today's new-lead list, so the model just formats — payload stays tiny
    regardless of CRM size.
    """
    today = _today_iso()

    user_map = {
        u["id"]: f"{u.get('firstName', '')} {u.get('lastName', '')}".strip()
        for u in data.get("users", [])
        if u.get("id")
    }

    def rep(uid: str) -> str:
        return user_map.get(uid, "") or "Unassigned"

    contacts = [
        {
            "name": f"{c.get('firstName', '')} {c.get('lastName', '')}".strip() or "Unnamed",
            "source": c.get("source") or "Unknown",
            "rep": rep(c.get("assignedTo", "")),
            "date": _fmt_date(c.get("dateAdded", "")),
        }
        for c in data.get("contacts", [])
    ]
    new_today = [c for c in contacts if c["date"] == today]

    opps = [
        {
            "value": o.get("monetaryValue", 0) or 0,
            "status": (o.get("status") or "").lower(),
            "rep": rep(o.get("assignedTo", "")),
            "updated": _fmt_date(o.get("updatedAt", "")),
        }
        for o in data.get("opportunities", [])
    ]
    open_opps = [o for o in opps if o["status"] == "open"]
    won = [o for o in opps if o["status"] == "won"]
    stale = [o for o in open_opps if (_days_since(o["updated"], today) or 0) > 5]

    top_rep = Counter(o["rep"] for o in open_opps).most_common(1)
    source_pool = new_today or contacts
    top_source = Counter(c["source"] for c in source_pool).most_common(1)

    summary = {
        "report_scope": "today" if new_today else "all-time (no contacts dated today)",
        "new_leads": len(new_today),
        "open_deals": len(open_opps),
        "pipeline_value": sum(o["value"] for o in open_opps),
        "won_deals": len(won),
        "stale_deals": len(stale),
        "top_rep": ({"name": top_rep[0][0], "open_deals": top_rep[0][1]} if top_rep else None),
        "top_source": (top_source[0][0] if top_source else None),
    }

    return {
        "summary": summary,
        "new_leads": [
            {"name": c["name"], "source": c["source"], "rep": c["rep"]}
            for c in new_today[:100]
        ],
    }


def _now() -> datetime:
    """Current time, in REPORT_TIMEZONE (IANA name) if set, else system local time."""
    tz_name = os.getenv("REPORT_TIMEZONE")
    if tz_name:
        try:
            from zoneinfo import ZoneInfo

            return datetime.now(ZoneInfo(tz_name))
        except Exception as e:  # bad name or missing tzdata — fall back to local
            print(f"[agent] REPORT_TIMEZONE={tz_name!r} unusable ({e}); using local time")
    return datetime.now()


def _report_clock() -> tuple[str, str]:
    """Return (header_date, footer_time), e.g. ('Monday, 16 Jun 2026', '09:01')."""
    now = _now()
    day = str(int(now.strftime("%d")))  # drop leading zero
    header_date = f"{now.strftime('%A')}, {day} {now.strftime('%b %Y')}"
    footer_time = now.strftime("%H:%M")
    return header_date, footer_time


def generate_report(data: dict) -> str:
    system = _system_prompt_path.read_text(encoding="utf-8")
    report_date, sent_time = _report_clock()
    user_message = (
        f"REPORT DATE (use verbatim in the header date line): {report_date}\n"
        f"SENT TIME (use verbatim in the footer timestamp): {sent_time}\n\n"
        "Here is today's CRM data. Generate the sales report:\n\n"
        + json.dumps(_slim_for_report(data), ensure_ascii=False, default=str)
    )
    try:
        response = _client.chat.completions.create(
            model=_REPORT_MODEL,
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
            model=_QA_MODEL,
            max_tokens=4000,
            messages=[
                {"role": "system", "content": _QA_SYSTEM},
                {"role": "user", "content": user_message},
            ],
        )
        return _clean_output(response.choices[0].message.content or "")
    except OpenAIError as e:
        raise RuntimeError(f"OpenAI error: {e}") from e
