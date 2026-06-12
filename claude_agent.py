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
    user_message = json.dumps(data, ensure_ascii=False, default=str)

    with _client.messages.stream(
        model="claude-opus-4-8",
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=system,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        message = stream.get_final_message()

    return next(
        (block.text for block in message.content if block.type == "text"),
        "",
    )
