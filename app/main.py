"""Routes and Server-Sent Events (SSE) streaming."""
import asyncio
import io
import json
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from pypdf import PdfReader
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app import agent, llm

ROOT = Path(__file__).resolve().parent.parent
MAX_SESSIONS = 50
MAX_TEXT_CHARS = 30_000
MAX_QUESTIONS = 10
MAX_QUESTION_CHARS = 1_000
MAX_MESSAGE_CHARS = 2_000
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
KEEPALIVE_SECONDS = 15
POLL_SECONDS = 0.2

app = FastAPI(title="Job Application Agent")
# Only answer requests addressed to this machine; blocks DNS-rebinding from other websites
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])

# In memory only: nothing is written to disk. Oldest sessions are dropped first.
sessions: "OrderedDict[str, agent.Session]" = OrderedDict()
sessions_lock = threading.Lock()


def new_session(session: agent.Session) -> str:
    session_id = uuid.uuid4().hex
    with sessions_lock:
        sessions[session_id] = session
        while len(sessions) > MAX_SESSIONS:
            sessions.popitem(last=False)
    return session_id


def get_session(session_id: str) -> agent.Session:
    with sessions_lock:
        session = sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "Unknown session. It may have expired; start a new run.")
    return session


def start_turn(session: agent.Session, message: str) -> None:
    """Run one agent turn in a background thread; the UI reads its events over SSE."""
    session.status = "running"  # set here too so a fast request never sees "idle"
    threading.Thread(target=agent.run_turn, args=(session, message), daemon=True).start()


# --- simple routes --------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/config")
def config():
    """What the UI shows in its header. Never includes keys."""
    model = "fake" if llm.fake_enabled() else llm.get_model()
    return {"model": model, "fake": llm.fake_enabled(), "provider": model.split("/")[0]}


@app.get("/api/sample")
def sample():
    samples = ROOT / "samples"
    read = lambda name: (samples / name).read_text(encoding="utf-8")  # noqa: E731
    return {"resume": read("resume.txt"), "job": read("job_post.txt"),
            "questions": [q for q in read("questions.txt").splitlines() if q.strip()]}


# --- run -------------------------------------------------------------------------

class RunRequest(BaseModel):
    resume: str
    job: str
    questions: list[str] | str = []  # a list, or one question per line


@app.post("/api/run")
def run(request: RunRequest):
    resume, job = request.resume.strip(), request.job.strip()
    questions = request.questions.splitlines() if isinstance(request.questions, str) else request.questions
    questions = [q.strip() for q in questions if q.strip()]

    if len(resume) < 50:
        raise HTTPException(400, "Your resume looks too short (at least 50 characters). Paste the full text or upload a file.")
    if len(job) < 10:
        raise HTTPException(400, "The job post looks too short (at least 10 characters). Paste the text or a link to it.")
    if len(resume) > MAX_TEXT_CHARS or len(job) > MAX_TEXT_CHARS:
        raise HTTPException(400, f"The resume and the job post must each be at most {MAX_TEXT_CHARS:,} characters.")
    if len(questions) > MAX_QUESTIONS:
        raise HTTPException(400, f"Please give at most {MAX_QUESTIONS} questions (one per line).")
    if any(len(q) > MAX_QUESTION_CHARS for q in questions):
        raise HTTPException(400, f"Each question must be at most {MAX_QUESTION_CHARS:,} characters.")
    try:
        llm.check_config()  # fail now with a clear message, e.g. a missing API key
    except llm.LLMConfigError as err:
        raise HTTPException(400, str(err))

    session = agent.Session(job_text=job, resume_text=resume, questions=questions)
    session_id = new_session(session)
    start_turn(session, agent.build_task_message(session))
    return {"session_id": session_id}


# --- stream ----------------------------------------------------------------------

def sse(data: dict, event_id: int | None = None) -> str:
    head = f"id: {event_id}\n" if event_id is not None else ""
    return f"{head}data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def event_stream(session: agent.Session, seen: int):
    """Yield events already logged, then new ones as they appear; end when the turn ends."""
    last_sent = time.monotonic()
    while True:
        running = session.status == "running"  # read before the events, so none are missed
        events = session.events_since(seen)
        for event in events:
            yield sse(event, event["id"])
        seen += len(events)
        if events:
            last_sent = time.monotonic()

        if not running:
            if session.status == "done":
                yield sse({"type": "done", "model_calls": session.model_calls})
            elif not events or events[-1]["type"] != "error":  # the agent's own error event is final
                yield sse({"type": "error", "message": "The run ended with an error."})
            return
        if time.monotonic() - last_sent > KEEPALIVE_SECONDS:
            yield ": keep-alive\n\n"
            last_sent = time.monotonic()
        await asyncio.sleep(POLL_SECONDS)


@app.get("/api/stream/{session_id}")
async def stream(session_id: str, after: int = 0, last_event_id: str | None = Header(default=None)):
    """`after` = how many events the client already has (browsers also send Last-Event-ID on reconnect)."""
    session = get_session(session_id)
    seen = max(after, 0)
    if last_event_id and last_event_id.isdigit():
        seen = max(seen, int(last_event_id) + 1)
    return StreamingResponse(event_stream(session, seen), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# --- chat (follow-up turns) -------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.post("/api/chat")
def chat(request: ChatRequest):
    session = get_session(request.session_id)
    message = request.message.strip()
    if session.status == "running":
        raise HTTPException(409, "The agent is still working. Wait for it to finish, then send your message.")
    if session.pack is None:
        raise HTTPException(400, "There is no application pack yet. Start a run first.")
    if not message or len(message) > MAX_MESSAGE_CHARS:
        raise HTTPException(400, f"Write a message of 1 to {MAX_MESSAGE_CHARS:,} characters.")
    after = len(session.events)  # the client streams from here to see only the new turn
    start_turn(session, message)
    return {"session_id": request.session_id, "after": after}


# --- resume upload ------------------------------------------------------------------

@app.post("/api/extract-resume")
async def extract_resume(file: UploadFile = File(...)):
    """Turn an uploaded resume into text. Works on bytes in memory; nothing is saved."""
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, "That file is larger than 5 MB. Paste the text of your resume instead.")
    name = (file.filename or "").lower()

    if name.endswith((".txt", ".md")):
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise HTTPException(400, "Could not read that file as text. Save it as UTF-8 or paste the text.")
    elif name.endswith(".pdf"):
        try:
            text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)
        except Exception:
            raise HTTPException(400, "Could not read that PDF. Paste the text of your resume instead.")
    else:
        raise HTTPException(400, "Unsupported file type. Upload a .txt, .md or .pdf file, or paste the text.")

    text = text.strip()
    if not text:
        raise HTTPException(400, "No text found in that file (a scanned PDF?). Paste the text of your resume instead.")
    return {"text": text, "filename": file.filename, "chars": len(text)}
