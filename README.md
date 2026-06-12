# hermes-sales-agent

AI-powered sales agent using Claude, GoHighLevel, Telegram, and GitHub.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in your keys in .env
```

## Project Structure

```
main.py             # Entry point
ghl_client.py       # GoHighLevel API client
claude_agent.py     # Claude AI agent logic
telegram_client.py  # Telegram bot client
github_client.py    # GitHub API client
prompts/            # System prompts
reports/            # Generated reports (gitignored)
```
