# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`hermes-sales-agent` is a Python AI sales agent for a GoHighLevel (GHL) CRM. It runs
laptop-independent: a Vercel serverless webhook handles live Telegram Q&A, and a GitHub
Action sends a deterministic daily report. The LLM is DeepSeek via OpenRouter (Q&A only;
the report is built in pure Python).

## Modules

- `api/webhook.py` — Vercel serverless Telegram webhook (Q&A + on-demand `/report`)
- `run_report.py` — daily report via GitHub Actions (fires once at 08:00 America/Chicago)
- `claude_agent.py` — LLM client + slimming/formatting (`_slim_for_qa`, `_slim_for_report`,
  `_format_report`, `_stage_maps`)
- `ghl_client.py` — GHL API client: paginated fetch, parallel fan-out, 429 retry/backoff
- `telegram_client.py` — Telegram send (Markdown→plain fallback), report splitting, archiving
- `github_client.py` — saves `reports/YYYY-MM-DD.md` via the GitHub Contents API
- `calendly_client.py` — Calendly KPIs (currently parked/disabled)

## Conventions

- Reports/answers focus on leads, new leads, pipelines, and pipeline stages — the team does
  NOT track monetary/deal value, so never report pipeline $ or deal amounts.
- Opportunity stage names are resolved from `pipelineStageId` via `_stage_maps` (the GHL
  search API returns IDs, not names).
- Exact CRM totals come from the search `meta.total`; individual record lists are the most
  recent N (the full CRM is too large to paginate live).
- NEVER commit `.env` (it is gitignored and holds all API keys).

## Environment

Secrets live in `.env` locally, and mirrored in Vercel env + GitHub Actions secrets:
`GHL_API_KEY`, `GHL_LOCATION_ID`, `GHL_COMPANY_ID`, `OPENROUTER_API_KEY` (or `OPENAI_API_KEY`),
`REPORT_MODEL`, `QA_MODEL`, `REPORT_TIMEZONE`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`,
`TELEGRAM_ADMIN_ID`, `GITHUB_TOKEN`, `GITHUB_OWNER`, `GITHUB_REPO`.

requirements.txt: openai, requests, python-dotenv, tzdata.
