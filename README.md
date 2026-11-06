# Job Application Agent

A local web app: paste a job post, your resume, and optional application questions. An AI agent works step by step (visibly) and produces an application pack: fit score, strengths, gaps, drafted answers, and a full cover letter. Fill in the highlighted blanks yourself, regenerate with your details, or chat to refine it. You can stop a run at any time.

Runs only on your machine (127.0.0.1:8000). The only data that leaves it goes to the model provider you choose; with Ollama, nothing does.

**Status:** skeleton, no features yet.

## Setup (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env   # then edit .env: pick LLM_MODEL and add that provider's key
```

## Run

```powershell
python run.py      # then open http://127.0.0.1:8000
python -m pytest   # run tests
```
