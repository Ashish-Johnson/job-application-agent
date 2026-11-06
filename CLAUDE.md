# Job Application Agent

## Product
The user pastes (1) a job post as text or URL, (2) their resume as text or PDF, (3) optional application questions, one per line. An AI agent works step by step, visibly, and produces an "application pack": a 0-100 fit score with breakdown, strengths backed by resume evidence, gaps with how to address each, a drafted answer to every question, and a short cover note. The user can then chat to request changes ("make answer 2 shorter").

## Hard constraints
- **Bring your own model:** all model calls go through LiteLLM (no provider SDKs). The user picks the model with `LLM_MODEL` in `.env` plus that provider's key. Supported: Gemini, OpenAI, Anthropic, Groq, OpenRouter, local Ollama (no key). Default is `gemini/gemini-2.5-flash` (free tier).
- **Hand-written agent loop:** manual tool calling so every step is visible and explainable in an interview. No agent frameworks.
- **Privacy-first, local only:** server bound to 127.0.0.1:8000, UI in the browser. No hosting, database, accounts, analytics, or storage outside the machine. The only data that leaves it is what is sent to the user's chosen model provider; nothing at all with Ollama.
- Python 3.11+, FastAPI + Uvicorn, ONE static HTML page with vanilla JS and CSS (no React, no build step).

## Honesty rule
The agent must never invent experience. Every claim about the candidate must be backed by resume text; missing facts become placeholders like `[ADD: ...]`.

## Code style
Keep it small and readable: plain functions, short files, comments only where the "why" isn't obvious. No abstractions that weren't asked for.

## Samples
`samples/` contains fictional data only. Never put real people's resumes or job posts there.

## Layout
- `app/main.py` - routes + Server-Sent Events
- `app/agent.py` - system prompt, Session, agent loop
- `app/tools.py` - five tools + schemas (OpenAI function-calling format)
- `app/llm.py` - LiteLLM wrapper; the only file that knows about providers
- `static/index.html` - the single page (vanilla JS/CSS)
- `samples/`, `tests/`, `run.py`, `README.md`, `requirements.txt`, `.env.example`

## How to run (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env   # set LLM_MODEL and the provider key
python run.py                 # http://127.0.0.1:8000
python -m pytest
```

## Workflow
**Always run tests before committing.**
