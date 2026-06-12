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
- Be concise. Get to the answer immediately.
- Use Telegram Markdown: *bold* for names/numbers, `code` for IDs/stages, _italic_ for labels.
- If the data does not contain enough information to answer, say so clearly.
- If asked about a specific person, deal, or contact, find and reference them by name."""


def generate_report(data: dict) -> str:
    system = _system_prompt_path.read_text(encoding="utf-8")
    user_message = (
        "Here is today's CRM data. Generate the sales report:\n\n"
        + json.dumps(data, ensure_ascii=False, default=str)
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
        + json.dumps(data, ensure_ascii=False, default=str)
        + f"\n\nQuestion: {question}"
    )
    try:
        response = _client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            messages=[
                {"role": "system", "content": _QA_SYSTEM},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content or ""
    except OpenAIError as e:
        return f"[Hermes] API error: {e}"
