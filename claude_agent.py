import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_system_prompt_path = Path(__file__).parent / "prompts" / "system_prompt.txt"

_QA_SYSTEM = """You are Hermes, an AI assistant embedded in a sales team's Telegram group.
You have access to live CRM data from GoHighLevel: contacts, opportunities, users, and pipelines.

OUTPUT MEDIUM: Your replies are sent via the Telegram Bot API with parse_mode=Markdown. \
Telegram renders triple-backtick fenced blocks as monospace code — ASCII tables display perfectly inside them. \
You have NO technical limitation on producing tables. Always output them directly when asked.

Rules:
- Answer only from the data provided. Never invent or estimate figures.
- NEVER truncate lists with "and X more" or "..." — always show every item in full.
- Give detailed, structured answers. Break down numbers, list names, show totals.
- If asked about leads: give the count, list names, sources, and who they're assigned to.
- If asked about deals: give count, total pipeline value, stage breakdown, and assigned reps.
- If asked about a rep: show all their deals, contacts, values, and stages.
- If asked about pipeline: show each stage with count and total value.
- Always end with a short *Summary* line with the key takeaway.
- Use Telegram Markdown: *bold* for names/numbers/totals, `code` for stages/IDs, _italic_ for labels.
- If the data does not contain enough information to answer, say so clearly."""

_TABLE_INSTRUCTION = """

Respond with a pipe-and-dash ASCII table inside a triple-backtick code fence. \
Do NOT add any explanation before the table — output it directly. \
Header row first, then a separator row of dashes, then one data row per item. Truncate names to 15 chars. \
Example:
```
| Name           | Deals | Value     |
|----------------|-------|-----------|
| John Smith     | 5     | $10,000   |
| Jane Doe       | 3     | $7,500    |
```"""


def _pick(record: dict, keys: list) -> dict:
    return {k: v for k, v in record.items() if k in keys and v not in (None, "", [], {})}


def _slim_for_qa(data: dict) -> dict:
    """Minimal context for Q&A — contacts, opportunities, users, pipeline names only."""
    return {
        "contacts": [
            _pick(c, ["id", "firstName", "lastName", "assignedTo", "source", "dateAdded", "tags"])
            for c in data.get("contacts", [])
        ],
        "opportunities": [
            _pick(o, ["id", "name", "monetaryValue", "status", "pipelineStageName",
                      "assignedTo", "updatedAt", "expectedCloseDate"])
            for o in data.get("opportunities", [])
        ],
        "users": [
            _pick(u, ["id", "firstName", "lastName"])
            for u in data.get("users", [])
        ],
        "pipelines": [
            {"name": p.get("name"),
             "stages": [s.get("name") for s in p.get("stages", [])]}
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


_TABLE_KEYWORDS = {"table", "tableau", "جدول", "tabela", "tabell"}


def answer_question(question: str, data: dict) -> str:
    wants_table = any(kw in question.lower() for kw in _TABLE_KEYWORDS)
    suffix = _TABLE_INSTRUCTION if wants_table else ""
    user_message = (
        "Here is the current CRM data:\n\n"
        + json.dumps(_slim_for_qa(data), ensure_ascii=False, default=str)
        + f"\n\nQuestion: {question}{suffix}"
    )
    response = _client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=4000,
        messages=[
            {"role": "system", "content": _QA_SYSTEM},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content or ""
