import json
import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
_system_prompt_path = Path(__file__).parent / "prompts" / "system_prompt.txt"


def generate_report(data: dict) -> str:
    system = _system_prompt_path.read_text(encoding="utf-8")
    user_message = (
        "Here is today's CRM data. Generate the sales report:\n\n"
        + json.dumps(data, ensure_ascii=False, default=str)
    )

    try:
        message = _client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        return next(
            (block.text for block in message.content if block.type == "text"),
            "",
        )
    except anthropic.APIError as e:
        return f"[Hermes] API error: {e}"
