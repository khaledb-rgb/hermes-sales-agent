import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_system_prompt_path = Path(__file__).parent / "prompts" / "system_prompt.txt"

_QA_SYSTEM = """You are Hermes, an AI assistant embedded in a sales team's Telegram group.
You have access to live CRM data from GoHighLevel: contacts, opportunities, users, pipelines, conversations, and invoices.

Rules:
- Answer only from the data provided. Never invent or estimate figures.
- Give detailed, structured answers. Break down numbers, list names, show totals.
- If asked about leads: give the count, list names, sources, and who they're assigned to.
- If asked about deals: give count, total pipeline value, stage breakdown, and assigned reps.
- If asked about a rep: show all their deals, contacts, values, and stages.
- If asked about pipeline: show each stage with count and total value.
- Always end with a short *Summary* line with the key takeaway.
- Use Telegram Markdown: *bold* for names/numbers/totals, `code` for stages/IDs, _italic_ for labels.
- If the data does not contain enough information to answer, say so clearly."""


def _slim(data: dict) -> dict:
    """Strip each record to essential fields only to stay within token limits."""

    def _pick(record: dict, keys: list) -> dict:
        return {k: v for k, v in record.items() if k in keys and v not in (None, "", [], {})}

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
            _pick(u, ["id", "firstName", "lastName", "email", "role"])
            for u in data.get("users", [])
        ],
        "pipelines": [
            {"id": p.get("id"), "name": p.get("name"),
             "stages": [{"id": s.get("id"), "name": s.get("name")} for s in p.get("stages", [])]}
            for p in data.get("pipelines", [])
        ],
        "conversations": [
            _pick(c, ["id", "contactId", "assignedTo", "lastMessageBody",
                      "lastMessageDate", "unreadCount"])
            for c in data.get("conversations", [])[:20]
        ],
        "invoices": [
            _pick(i, ["id", "status", "total", "dueDate", "contactId"])
            for i in data.get("invoices", [])
        ],
    }


def generate_report(data: dict) -> str:
    system = _system_prompt_path.read_text(encoding="utf-8")
    user_message = (
        "Here is today's CRM data. Generate the sales report:\n\n"
        + json.dumps(_slim(data), ensure_ascii=False, default=str)
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


def answer_question(question: str, data: dict) -> str:
    user_message = (
        "Here is the current CRM data:\n\n"
        + json.dumps(_slim(data), ensure_ascii=False, default=str)
        + f"\n\nQuestion: {question}"
    )
    try:
        response = _client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=1024,
            messages=[
                {"role": "system", "content": _QA_SYSTEM},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content or ""
    except OpenAIError as e:
        return f"[Hermes] API error: {e}"
