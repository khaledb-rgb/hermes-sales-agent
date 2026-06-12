import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_system_prompt_path = Path(__file__).parent / "prompts" / "system_prompt.txt"


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
