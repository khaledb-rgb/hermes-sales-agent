import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

import calendly_client

load_dotenv()

# LLM provider chain — tried in order at request time, so a rate-limit/credit
# error (429/402) on one provider automatically falls through to the next.
# Gemini is primary (free tier can 429); OpenAI is the safety net. Both speak
# the OpenAI chat-completions API; Gemini just needs a custom base_url.
def _build_providers() -> list:
    providers = []
    gem = os.getenv("GEMINI_API_KEY")
    if gem:
        providers.append((
            "gemini",
            OpenAI(api_key=gem, base_url=os.getenv(
                "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")),
            os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        ))
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        providers.append((
            "openai",
            OpenAI(api_key=openai_key),
            os.getenv("QA_MODEL", "gpt-4o-mini"),
        ))
    return providers


_PROVIDERS = _build_providers()

_system_prompt_path = Path(__file__).parent / "prompts" / "system_prompt.txt"

_QA_SYSTEM = """You are Hermes, an AI assistant embedded in a sales team's Telegram group.
You have access to live CRM data from GoHighLevel: contacts, opportunities, users, and pipelines.

The data has two parts:
- "summary": pre-computed aggregates. Read summary.coverage carefully — it tells you
  exactly what is exact vs. partial.
  • summary.*.total = EXACT whole-CRM counts. Use these for "how many total…" questions.
  • summary.*.loaded = how many individual records were actually loaded this request.
  • All breakdowns (by_source, by_stage, by_pipeline, by_rep) are computed ONLY over the
    loaded (most-recent) records — they are a recent-records view, NOT the whole CRM.
- "contacts" / "opportunities": the individual loaded records (most recent first). Use these
  to list names and look up details. They do NOT contain older records.
- "conversations": recent message threads (contact, lastMessage date, type, direction, unread
  count). Use for "what did we last say to / hear from X" and unread/follow-up questions.
- "appointments": calendar events from yesterday through ~3 weeks ahead (title, start as
  'YYYY-MM-DD HH:MM', status, rep). Use for "who's booked today/this week". Keep the start
  time exactly as given — do not drop the time.
- "invoices": invoices when present (often empty). Use for billing/payment questions.

OUTPUT MEDIUM: Your replies are sent via the Telegram Bot API with parse_mode=Markdown. \
Telegram renders triple-backtick fenced blocks as monospace code — ASCII tables display perfectly inside them. \
You have NO technical limitation on producing tables. Always output them directly when asked.

Rules:
- Answer only from the data provided. Never invent or estimate figures.
- For "how many total" questions, use summary.*.total (exact). For breakdowns/lists, use the
  loaded records and SAY they reflect the most-recent records loaded, not the whole CRM, when
  total > loaded.
- If asked for a specific person/deal and it is NOT in the loaded records, do NOT say it
  doesn't exist. Say it isn't among the most-recent records loaded and may be older — offer
  what you can (e.g. the exact total, or matching loaded records).
- Contacts include email, phone, and company when available — use them for lookup questions.
- NEVER truncate lists with "and X more" or "..." — always show every item in full.
- Give detailed, structured answers. Break down numbers, list names, show totals.
- DEFAULT list format: plain numbered list (1. *Name* — Source — Rep), NOT a code block table.
  Code block tables are only for when the user explicitly asks for a "table".
  Plain text lists scroll naturally in Telegram; code blocks have a fixed scroll area.
- We do NOT track deal/monetary value — never report pipeline $ or deal amounts; focus on
  counts, leads, pipelines, and stages.
- If asked about leads: give the count (from summary), then list each contact as a numbered line.
- If asked about deals: give count, stage breakdown, and assigned reps (no dollar values).
- If asked about a rep: show all their deals, contacts, and stages.
- "Stale" leads/deals = OPEN opportunities not updated recently. Each opportunity has a
  `daysSinceUpdate` field — use it. Default threshold is >5 days when the user doesn't give
  one; if they say "stale" with a number (e.g. "2 days"), use that. summary.opportunities.stale
  is the >5-day count. Never claim staleness can't be determined — daysSinceUpdate is always there.
- Each opportunity has a `pipeline` and a `stage` field. When asked about a specific \
pipeline (e.g. "New Sales Pipeline"), filter opportunities by their `pipeline` value.
- If asked about pipeline: show each stage with its count (use summary.opportunities.by_stage
  and by_pipeline).
- If asked about messages/conversations: use "conversations" — show contact, last message date,
  direction (inbound/outbound), and flag unread threads for follow-up.
- If asked about appointments/bookings: use "appointments" — list title, date+time, rep, and
  status; for "today" filter start to today's date.
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


def _stage_maps(pipelines: list) -> tuple[dict, dict]:
    """Build {stageId: stageName} and {stageId: pipelineName} from pipeline defs.

    The GHL /opportunities/search endpoint returns pipelineStageId / pipelineId
    (UUIDs), NOT human-readable names. Without these maps every opportunity's
    'stage' resolves to blank, which is why no stage/pipeline ever showed up.
    """
    stage_name, stage_pipeline = {}, {}
    for p in pipelines:
        pname = p.get("name", "")
        for s in p.get("stages", []):
            sid = s.get("id")
            if sid:
                stage_name[sid] = s.get("name", "")
                stage_pipeline[sid] = pname
    return stage_name, stage_pipeline


def _epoch_to_dt(val) -> datetime | None:
    """Epoch seconds or milliseconds → UTC datetime. GHL returns some dates
    (e.g. conversations.lastMessageDate) as epoch ms ints, not ISO strings."""
    try:
        ts = float(val)
        ts = ts / 1000 if ts > 1e11 else ts  # treat large values as milliseconds
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def _fmt_date(val) -> str:
    """2026-06-12T16:14:28.130Z → 2026-06-12 (also accepts epoch ms ints)."""
    if isinstance(val, (int, float)):
        dt = _epoch_to_dt(val)
        return dt.strftime("%Y-%m-%d") if dt else ""
    return val[:10] if val and "T" in val else (val or "")


def _fmt_dt(val) -> str:
    """2026-06-12T16:14:28Z → '2026-06-12 16:14' (also accepts epoch ms ints).

    Uses a space (not 'T') on purpose: _clean_output strips the time off any
    'T'-form ISO timestamp the model echoes, which would erase appointment
    times. The space form survives that pass.
    """
    if isinstance(val, (int, float)):
        dt = _epoch_to_dt(val)
        return dt.strftime("%Y-%m-%d %H:%M") if dt else ""
    if not val or "T" not in val:
        return val or ""
    d, t = val.split("T", 1)
    return f"{d} {t[:5]}"


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

    stage_name, stage_pipeline = _stage_maps(data.get("pipelines", []))
    today = _today_iso()

    # Build full contact list (slim fields) for summary stats
    all_contacts = [
        {
            "name": f"{c.get('firstName', '')} {c.get('lastName', '')}".strip(),
            "source": c.get("source", "") or "Unknown",
            "email": c.get("email", "") or "",
            "phone": c.get("phone", "") or "",
            "company": c.get("companyName", "") or "",
            "assignedTo": rep(c.get("assignedTo", "")),
            "dateAdded": _fmt_date(c.get("dateAdded", "")),
            "tags": c.get("tags", []),
        }
        for c in data.get("contacts", [])
    ]

    all_opps = [
        {
            "name": o.get("name", ""),
            "status": o.get("status", ""),
            "stage": stage_name.get(o.get("pipelineStageId", ""), ""),
            "pipeline": stage_pipeline.get(o.get("pipelineStageId", ""), ""),
            "assignedTo": rep(o.get("assignedTo", "")),
            "updatedAt": _fmt_date(o.get("updatedAt", "")),
            "daysSinceUpdate": _days_since(_fmt_date(o.get("updatedAt", "")), today),
            "expectedCloseDate": _fmt_date(o.get("expectedCloseDate", "")),
        }
        for o in data.get("opportunities", [])
    ]

    # Summary stats — computed over ALL records
    contact_by_source = dict(Counter(c["source"] for c in all_contacts).most_common())
    contact_by_rep = dict(Counter(c["assignedTo"] for c in all_contacts).most_common())

    open_opps = [o for o in all_opps if o["status"] == "open"]
    opp_by_stage = dict(Counter(o["stage"] for o in open_opps if o["stage"]).most_common())
    opp_by_pipeline = dict(Counter(o["pipeline"] for o in open_opps if o["pipeline"]).most_common())
    opp_by_rep = dict(Counter(o["assignedTo"] for o in open_opps).most_common())
    stale_opps = [o for o in open_opps if (o["daysSinceUpdate"] or 0) > 5]

    # Exact whole-CRM totals come from the search meta (data.*_total). The
    # breakdowns/lists below are computed only over the most-recent records
    # actually loaded this request, so we expose both numbers and a note.
    contacts_total = data.get("contacts_total") or len(all_contacts)
    opps_total = data.get("opportunities_total") or len(all_opps)

    summary = {
        "coverage": (
            f"`total` fields are EXACT whole-CRM counts. All breakdowns "
            f"(by_source, by_stage, by_pipeline, by_rep) and the individual "
            f"contact/opportunity lists are computed from only the {len(all_contacts)} "
            f"most-recent contacts and {len(all_opps)} most-recent opportunities "
            f"loaded this request - older records are not in those lists."
        ),
        "contacts": {
            "total": contacts_total,
            "loaded": len(all_contacts),
            "by_source": contact_by_source,
            "by_rep": contact_by_rep,
        },
        "opportunities": {
            "total": opps_total,
            "loaded": len(all_opps),
            "open": len(open_opps),
            "stale": len(stale_opps),
            "by_pipeline": opp_by_pipeline,
            "by_stage": opp_by_stage,
            "by_rep": opp_by_rep,
        },
    }

    # Most recent individual records (sorted by date descending)
    recent_contacts = sorted(
        all_contacts, key=lambda c: c["dateAdded"], reverse=True
    )[:300]

    recent_opps = sorted(
        all_opps, key=lambda o: o["updatedAt"], reverse=True
    )[:200]

    # Conversations — most recent message threads (for "what did we last say to X")
    conversations = sorted(
        (
            {
                "contact": c.get("fullName") or c.get("contactName") or "",
                "company": c.get("companyName", "") or "",
                "lastMessage": _fmt_date(c.get("lastMessageDate", "")),
                "type": c.get("lastMessageType", "") or "",
                "direction": c.get("lastMessageDirection", "") or "",
                "unread": c.get("unreadCount", 0) or 0,
            }
            for c in data.get("conversations", [])
        ),
        key=lambda c: c["lastMessage"],
        reverse=True,
    )

    # Appointments — calendar events in the fetch window, soonest first.
    today = _today_iso()
    appointments = sorted(
        (
            {
                "title": e.get("title", "") or "",
                "start": _fmt_dt(e.get("startTime", "")),
                "status": e.get("appointmentStatus") or e.get("appoinmentStatus") or "",
                "rep": rep(e.get("assignedUserId", "")),
            }
            for e in data.get("appointments", [])
        ),
        key=lambda a: a["start"],
    )

    # Invoices — best-effort field mapping (shape unverified; usually empty here).
    invoices = [
        {
            "name": i.get("name") or i.get("title") or i.get("invoiceNumber") or "",
            "status": i.get("status", "") or "",
            "total": i.get("total", 0) or i.get("amountDue", 0) or 0,
            "contact": (i.get("contactDetails") or {}).get("name", "")
            or i.get("contactName", "")
            or "",
            "due": _fmt_date(i.get("dueDate", "")),
        }
        for i in data.get("invoices", [])
    ]

    summary["conversations"] = {
        "loaded": len(conversations),
        "unread_threads": sum(1 for c in conversations if c["unread"]),
    }
    summary["appointments"] = {
        "loaded": len(appointments),
        "upcoming": sum(1 for a in appointments if a["start"][:10] >= today),
        "window": "yesterday through ~3 weeks ahead",
    }
    summary["invoices"] = {"loaded": len(invoices)}

    return {
        "summary": summary,
        "contacts": recent_contacts,
        "opportunities": recent_opps,
        "conversations": conversations,
        "appointments": appointments,
        "invoices": invoices,
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


# Characters that break Telegram's legacy Markdown parser when they appear in
# data values (names, sources). Neutralized so the report renders cleanly.
_MD_BREAK = str.maketrans({"_": " ", "*": " ", "`": "'", "[": "(", "]": ")"})


def _safe(s: str) -> str:
    return (s or "").translate(_MD_BREAK).strip()


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
        return _safe(user_map.get(uid, "")) or "Unassigned"

    contacts = [
        {
            "name": _safe(f"{c.get('firstName', '')} {c.get('lastName', '')}".strip()) or "Unnamed",
            "source": _safe(c.get("source")) or "Unknown",
            "rep": rep(c.get("assignedTo", "")),
            "date": _fmt_date(c.get("dateAdded", "")),
        }
        for c in data.get("contacts", [])
    ]
    new_today = [c for c in contacts if c["date"] == today]
    # dedup identical contacts (CRM sometimes has repeats) by name/source/rep
    seen, deduped = set(), []
    for c in new_today:
        key = (c["name"], c["source"], c["rep"])
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    new_today = deduped

    stage_name, stage_pipeline = _stage_maps(data.get("pipelines", []))

    opps = [
        {
            "status": (o.get("status") or "").lower(),
            "rep": rep(o.get("assignedTo", "")),
            "updated": _fmt_date(o.get("updatedAt", "")),
            "stage": _safe(stage_name.get(o.get("pipelineStageId", ""), "")),
            "pipeline": _safe(stage_pipeline.get(o.get("pipelineStageId", ""), "")),
        }
        for o in data.get("opportunities", [])
    ]
    open_opps = [o for o in opps if o["status"] == "open"]
    won = [o for o in opps if o["status"] == "won"]
    stale = [o for o in open_opps if (_days_since(o["updated"], today) or 0) > 5]

    by_pipeline = [
        {"name": n, "count": c}
        for n, c in Counter(
            o["pipeline"] for o in open_opps if o["pipeline"]
        ).most_common(_PIPELINES_SHOWN)
    ]
    by_stage = [
        {"name": n, "count": c}
        for n, c in Counter(
            o["stage"] for o in open_opps if o["stage"]
        ).most_common(_STAGES_SHOWN)
    ]

    top_rep = Counter(o["rep"] for o in open_opps if o["rep"] != "Unassigned").most_common(1)
    source_pool = new_today or contacts
    top_source = Counter(c["source"] for c in source_pool).most_common(1)

    summary = {
        "report_scope": "today" if new_today else "all-time (no contacts dated today)",
        "new_leads": len(new_today),
        "open_deals": len(open_opps),
        "by_pipeline": by_pipeline,
        "by_stage": by_stage,
        "won_deals": len(won),
        "stale_deals": len(stale),
        "top_rep": ({"name": top_rep[0][0], "open_deals": top_rep[0][1]} if top_rep else None),
        "top_source": (top_source[0][0] if top_source else None),
    }

    return {
        "summary": summary,
        "new_leads": [
            {"name": c["name"], "source": c["source"], "rep": c["rep"]}
            for c in new_today[:_LEADS_SHOWN]
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


_LEADS_SHOWN = 40  # cap the new-leads list; show "+N more" beyond this
_PIPELINES_SHOWN = 8  # cap the open-deals-by-pipeline breakdown
_STAGES_SHOWN = 10  # cap the open-deals-by-stage breakdown


def _format_report(slim: dict, report_date: str, sent_time: str) -> str:
    """Build the two-message Telegram report deterministically from pre-computed
    data. No LLM — the template is fixed and the figures are already computed, so
    this is exact, properly escaped, and free of the duplication/Markdown issues
    an LLM introduces on long data lists.
    """
    s = slim["summary"]
    leads = slim["new_leads"]
    total_new = s["new_leads"]
    footer = f"_Sent by Hermes · {sent_time}_"

    # --- Message 1: KPIs ---
    # With Calendly data, lead with Bookings | New leads | Show rate and demote
    # Open deals to a data row; otherwise use the CRM-only New leads | Open deals row.
    if "bookings_today" in s:
        title = "📊 *Daily CRM + Calendly Report*"
        show = f"{s['show_rate']}%" if s.get("show_rate") is not None else "—"
        msg1 = [title, f"_{report_date}_", "", "———", ""] + [
            "`Bookings` | `New leads` | `Show rate`",
            f"   {s['bookings_today']} | {total_new} | {show}",
        ]
        rows = [f"• Open deals: {s['open_deals']}"]
    else:
        title = "📊 *Daily CRM Report*"
        msg1 = [title, f"_{report_date}_", "", "———", ""] + [
            "`New leads` | `Open deals`",
            f"   {total_new} | {s['open_deals']}",
        ]
        rows = []

    # Pipeline + stage breakdowns (open deals).
    if s.get("by_pipeline"):
        msg1 += ["", "———", "", "*Open deals by pipeline:*"]
        msg1 += [f"• {p['name']} — {p['count']}" for p in s["by_pipeline"]]
    if s.get("by_stage"):
        msg1 += ["", "———", "", "*Top stages:*"]
        msg1 += [f"• {st['name']} — {st['count']}" for st in s["by_stage"]]

    if s.get("top_rep"):
        rows.append(f"• Top rep: *{s['top_rep']['name']}* — {s['top_rep']['open_deals']} deals")
    if s.get("top_source"):
        rows.append(f"• Top lead source: *{s['top_source']}*")
    if rows:
        msg1 += ["", "———", ""] + rows

    tags = []
    if s.get("won_deals"):
        tags.append(f"#closed_{s['won_deals']}")
    if s.get("stale_deals"):
        tags.append(f"#followup_{s['stale_deals']}")
    if total_new:
        tags.append(f"#new_{total_new}")
    if tags:
        # Escape the underscores so Telegram's Markdown doesn't read them as
        # italics (#new_43 would otherwise open an unclosed entity and 400).
        msg1 += ["", "  ".join(tags).replace("_", "\\_")]
    msg1 += ["", footer]

    # --- Message 2: new leads ---
    if leads:
        lines = [
            f"{i}. *{l['name']}* — {l['source']} — {l['rep']}"
            for i, l in enumerate(leads, 1)
        ]
        if total_new > len(leads):
            lines.append(f"_+{total_new - len(leads)} more_")
        body = "\n".join(lines)
    else:
        body = "_No new leads today._"

    msg2 = [
        "🧲 *New Leads*",
        f"_{report_date}_",
        "",
        "———",
        "",
        body,
        "",
        footer,
    ]

    return "\n".join(msg1) + "\n---SPLIT---\n" + "\n".join(msg2)


def generate_report(data: dict, include_calendly: bool = True) -> str:
    """Build the daily report. include_calendly adds Bookings/Show-rate KPIs but
    costs extra Calendly API calls — the daily GitHub Action keeps it on; the
    interactive webhook /report turns it off so group replies stay fast."""
    slim = _slim_for_report(data)
    if include_calendly:
        try:
            cal = calendly_client.fetch_summary()
            if cal:
                slim["summary"].update(cal)
        except Exception as e:  # Calendly is best-effort — never block the report
            print(f"[agent] calendly summary skipped: {e}")
    report_date, sent_time = _report_clock()
    return _format_report(slim, report_date, sent_time)


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


def answer_question(question: str, data: dict, history: list | None = None) -> str:
    """Answer a CRM question.

    history (optional) is a list of prior {role, content} turns for this chat —
    plain text only, no CRM data — so the model can resolve follow-ups like
    "what about his email?" or "which of those are stale?". The current question
    always carries fresh CRM data; prior turns give conversational continuity.
    """
    wants_table = any(kw in question.lower() for kw in _TABLE_KEYWORDS)
    suffix = _TABLE_INSTRUCTION if wants_table else ""
    user_message = (
        "Here is the current CRM data:\n\n"
        + json.dumps(_slim_for_qa(data), ensure_ascii=False, default=str)
        + f"\n\nQuestion: {question}{suffix}"
    )
    if not _PROVIDERS:
        raise RuntimeError("No LLM provider configured (set GEMINI_API_KEY or OPENAI_API_KEY).")

    messages = [{"role": "system", "content": _QA_SYSTEM}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    errors = []
    for name, client, model in _PROVIDERS:
        try:
            response = client.chat.completions.create(
                model=model, max_tokens=4000, messages=messages,
            )
            return _clean_output(response.choices[0].message.content or "")
        except OpenAIError as e:
            print(f"[agent] {name} ({model}) failed, trying next provider: {e}")
            errors.append(f"{name}: {e}")
    raise RuntimeError("All LLM providers failed -> " + " | ".join(errors))
