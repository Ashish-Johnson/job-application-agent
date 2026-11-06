import io
import json
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app import agent, main

# The scripted fake model is written for the sample files, so runs use them
SAMPLES = Path(__file__).parent.parent / "samples"
RESUME = (SAMPLES / "resume.txt").read_text(encoding="utf-8")
JOB = (SAMPLES / "job_post.txt").read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    monkeypatch.setenv("FAKE_LLM", "1")
    main.sessions.clear()


@pytest.fixture
def client():
    return TestClient(main.app, base_url="http://127.0.0.1:8000")


def start_run(client, **overrides):
    questions = (SAMPLES / "questions.txt").read_text(encoding="utf-8").splitlines()
    body = {"resume": RESUME, "job": JOB, "questions": questions, **overrides}
    return client.post("/api/run", json=body)


def read_stream(client, session_id, after=0):
    """Collect (event dicts, raw text) from an SSE response until the server closes it."""
    with client.stream("GET", f"/api/stream/{session_id}", params={"after": after}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        raw = "".join(response.iter_text())
    events = [json.loads(line[len("data: "):]) for line in raw.splitlines() if line.startswith("data: ")]
    return events, raw


# --- simple routes ----------------------------------------------------------------

def test_index_serves_the_page(client):
    response = client.get("/")
    assert response.status_code == 200 and "<title>" in response.text


def test_config_never_leaks_keys(client, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-value")
    data = client.get("/api/config").json()
    assert data == {"model": "fake", "fake": True, "provider": "fake"}
    assert "super-secret-value" not in client.get("/api/config").text


def test_config_for_a_real_model(client, monkeypatch):
    monkeypatch.delenv("FAKE_LLM")
    monkeypatch.setenv("LLM_MODEL", "groq/llama-3.3-70b")
    assert client.get("/api/config").json() == {
        "model": "groq/llama-3.3-70b", "fake": False, "provider": "groq"}


def test_sample(client):
    data = client.get("/api/sample").json()
    assert "MAYA" in data["resume"] and "AI Builder" in data["job"] and len(data["questions"]) == 3


def test_other_host_headers_are_refused(client):
    assert client.get("/api/config", headers={"host": "evil.example.com"}).status_code == 400


# --- validation ---------------------------------------------------------------------

@pytest.mark.parametrize("overrides, expected", [
    ({"resume": "too short"}, "resume looks too short"),
    ({"job": "hi"}, "job post looks too short"),
    ({"resume": "x" * 30_001}, "at most 30,000"),
    ({"job": "x" * 30_001}, "at most 30,000"),
    ({"questions": [f"Question {i}?" for i in range(11)]}, "at most 10 questions"),
    ({"questions": ["x" * 1_001]}, "at most 1,000"),
])
def test_run_validation_errors(client, overrides, expected):
    response = start_run(client, **overrides)
    assert response.status_code == 400 and expected in response.json()["detail"]
    assert len(main.sessions) == 0


def test_run_reports_a_missing_api_key_up_front(client, monkeypatch):
    monkeypatch.delenv("FAKE_LLM")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-4o-mini")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = start_run(client)
    assert response.status_code == 400 and "OPENAI_API_KEY" in response.json()["detail"]


def test_questions_can_be_one_string_per_line(client):
    response = start_run(client, questions="First?\n\n  Second?  \n")
    assert response.status_code == 200
    assert main.sessions[response.json()["session_id"]].questions == ["First?", "Second?"]


# --- a full run over SSE ---------------------------------------------------------------

def test_full_run_streams_to_done_with_a_pack(client):
    session_id = start_run(client).json()["session_id"]
    events, raw = read_stream(client, session_id)

    types = [e["type"] for e in events]
    assert types[-1] == "done" and events[-1]["model_calls"] == 3
    assert types.count("tool_start") == 3 and "result" in types and "error" not in types
    pack = next(e for e in events if e["type"] == "result")["pack"]
    assert pack["score"] == main.sessions[session_id].fit["score"] and pack["answers"]
    assert "id: 0\n" in raw  # events carry ids so browsers can resume after a reconnect


def test_a_finished_run_can_be_replayed_and_resumed_midway(client):
    session_id = start_run(client).json()["session_id"]
    first, _ = read_stream(client, session_id)
    again, _ = read_stream(client, session_id)
    assert [e["type"] for e in again] == [e["type"] for e in first]  # full replay

    rest, _ = read_stream(client, session_id, after=5)
    assert [e["type"] for e in rest] == [e["type"] for e in first][5:]


def test_stream_ends_with_error_when_the_run_fails(client, monkeypatch):
    def failing_turn(session, message):
        session.add_event("error", message="boom")
        session.status = "error"
    monkeypatch.setattr(main.agent, "run_turn", failing_turn)
    events, _ = read_stream(client, start_run(client).json()["session_id"])
    assert [e["type"] for e in events] == ["error"] and events[0]["message"] == "boom"


def test_stream_adds_an_error_event_if_the_agent_did_not(client, monkeypatch):
    monkeypatch.setattr(main.agent, "run_turn", lambda session, message: setattr(session, "status", "error"))
    events, _ = read_stream(client, start_run(client).json()["session_id"])
    assert events[-1]["type"] == "error"


# --- follow-up chat -----------------------------------------------------------------------

def test_follow_up_chat_changes_the_pack(client):
    session_id = start_run(client).json()["session_id"]
    first, _ = read_stream(client, session_id)
    before = next(e for e in first if e["type"] == "result")["pack"]

    reply = client.post("/api/chat", json={"session_id": session_id, "message": "Please make answer 1 shorter."})
    assert reply.status_code == 200 and reply.json()["after"] == len(first) - 1  # events minus the "done" marker

    events, _ = read_stream(client, session_id, after=reply.json()["after"])
    assert [e["type"] for e in events][-2:] == ["result", "done"]
    after = next(e for e in events if e["type"] == "result")["pack"]
    assert len(after["answers"][0]["answer"].split()) < len(before["answers"][0]["answer"].split())


def test_follow_up_question_gets_a_reply_event(client):
    session_id = start_run(client).json()["session_id"]
    first, _ = read_stream(client, session_id)
    after = client.post("/api/chat", json={"session_id": session_id, "message": "What is my biggest gap?"}).json()["after"]
    events, _ = read_stream(client, session_id, after=after)
    assert [e["type"] for e in events] == ["reply", "done"]


def test_chat_while_running_is_409(client, monkeypatch):
    release = threading.Event()

    def slow_turn(session, message):
        release.wait(timeout=10)
        session.status = "done"
    monkeypatch.setattr(main.agent, "run_turn", slow_turn)
    try:
        session_id = start_run(client).json()["session_id"]
        response = client.post("/api/chat", json={"session_id": session_id, "message": "hello"})
        assert response.status_code == 409 and "still working" in response.json()["detail"]
    finally:
        release.set()


def test_chat_before_there_is_a_pack_is_400(client):
    session_id = main.new_session(agent.Session(status="done"))
    response = client.post("/api/chat", json={"session_id": session_id, "message": "hello"})
    assert response.status_code == 400 and "no application pack" in response.json()["detail"]


def test_unknown_session_is_404(client):
    assert client.get("/api/stream/nope").status_code == 404
    assert client.post("/api/chat", json={"session_id": "nope", "message": "hi"}).status_code == 404


# --- sessions are capped ----------------------------------------------------------------------

def test_oldest_sessions_are_dropped(monkeypatch):
    monkeypatch.setattr(main, "MAX_SESSIONS", 3)
    ids = [main.new_session(agent.Session()) for _ in range(5)]
    assert list(main.sessions) == ids[-3:]


# --- resume upload ------------------------------------------------------------------------------

def make_pdf(text: str) -> bytes:
    """A tiny one-page PDF containing `text`, built by hand."""
    stream = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 200] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = b"%PDF-1.4\n", []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (len(objects) + 1, xref)
    return out


def upload(client, name, data):
    return client.post("/api/extract-resume", files={"file": (name, data)})


@pytest.mark.parametrize("name", ["resume.txt", "resume.md"])
def test_extract_text_files(client, name):
    response = upload(client, name, "Maya\nPython, FastAPI\n".encode())
    assert response.status_code == 200 and response.json()["text"] == "Maya\nPython, FastAPI"


def test_extract_pdf(client):
    response = upload(client, "cv.pdf", make_pdf("Python and FastAPI developer"))
    assert response.status_code == 200 and "Python and FastAPI developer" in response.json()["text"]


def test_extract_rejects_bad_uploads(client):
    blank = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(blank)

    cases = [
        (upload(client, "resume.docx", b"x"), "Unsupported file type"),
        (upload(client, "resume.txt", b"x" * (5 * 1024 * 1024 + 1)), "larger than 5 MB"),
        (upload(client, "resume.pdf", b"this is not a pdf"), "Could not read that PDF"),
        (upload(client, "resume.pdf", blank.getvalue()), "No text found"),
        (upload(client, "resume.txt", b"\xff\xfe\x00\x01 bad"), "Could not read that file as text"),
        (upload(client, "resume.txt", b"   \n"), "No text found"),
    ]
    for response, expected in cases:
        assert response.status_code == 400 and expected in response.json()["detail"], expected


# --- the page itself ----------------------------------------------------------------------

def test_index_page_is_the_ui(client):
    page = client.get("/").text
    for needle in ['id="btn-run"', "Run agent", 'id="timeline"', 'id="results"', 'id="chat-form"',
                   "Use sample data", "Upload PDF/TXT", "/api/stream/", "/api/extract-resume"]:
        assert needle in page, needle
    # Privacy: the page must never load anything from the internet (no CDNs, fonts, analytics)
    assert "http://" not in page and "https://" not in page
    # Safety: model and user text is only ever inserted as text, never parsed as HTML
    assert "innerHTML" not in page and "insertAdjacentHTML" not in page and "document.write" not in page


# --- stop -------------------------------------------------------------------------------------------------

def slow_model(monkeypatch):
    """Make model calls hang until released. Returns (entered, release, original_chat)."""
    entered, release = threading.Event(), threading.Event()
    original = main.agent.llm.chat

    def hanging_chat(system, history, tools=None, on_wait=None, tool_choice=None):
        entered.set()
        release.wait(timeout=10)
        return original(system, history, tools, on_wait=on_wait, tool_choice=tool_choice)

    monkeypatch.setattr(main.agent.llm, "chat", hanging_chat)
    return entered, release, original


def test_stop_unknown_session_is_404_and_idle_session_is_409(client):
    assert client.post("/api/stop/nope").status_code == 404
    session_id = main.new_session(agent.Session(status="done"))
    response = client.post(f"/api/stop/{session_id}")
    assert response.status_code == 409 and "Nothing is running" in response.json()["detail"]


def test_stop_ends_the_stream_with_stopped_then_done(client, monkeypatch):
    entered, release, _ = slow_model(monkeypatch)
    try:
        session_id = start_run(client).json()["session_id"]
        assert entered.wait(5)
        assert client.post(f"/api/stop/{session_id}").json() == {"ok": True}
        events, _ = read_stream(client, session_id)
    finally:
        release.set()
    assert [e["type"] for e in events] == ["stopped", "done"]
    assert events[0]["message"] == "Stopped by you" and events[1]["status"] == "stopped"
    assert main.sessions[session_id].status == "stopped"


def test_followups_still_work_after_a_stopped_turn_and_the_pack_is_kept(client, monkeypatch):
    session_id = start_run(client).json()["session_id"]
    read_stream(client, session_id)
    pack = main.sessions[session_id].pack

    entered, release, original = slow_model(monkeypatch)
    try:
        client.post("/api/chat", json={"session_id": session_id, "message": "Please make answer 2 shorter."})
        assert entered.wait(5)
        assert client.post(f"/api/stop/{session_id}").status_code == 200
        read_stream(client, session_id, after=len(main.sessions[session_id].events) - 1)
    finally:
        release.set()
    assert main.sessions[session_id].status == "stopped" and main.sessions[session_id].pack is pack

    monkeypatch.setattr(main.agent.llm, "chat", original)  # the model is fast again
    after = client.post("/api/chat", json={"session_id": session_id, "message": "What is my biggest gap?"})
    assert after.status_code == 200
    events, _ = read_stream(client, session_id, after=after.json()["after"])
    assert [e["type"] for e in events] == ["reply", "done"]


# --- regenerate with my details ------------------------------------------------------------------------------

import re  # noqa: E402

SCORE_FIELDS = ("score", "verdict", "missing_must_haves", "breakdown")


def finished_run(client):
    session_id = start_run(client).json()["session_id"]
    events, _ = read_stream(client, session_id)
    return session_id, next(e for e in events if e["type"] == "result")["pack"]


def hints_in(text):
    return [h.strip() for h in re.findall(r"\[ADD:\s*(.*?)\]", text)]


def regenerate(client, session_id, facts):
    return client.post("/api/regenerate", json={"session_id": session_id, "facts": facts})


def test_regenerate_rewrites_with_the_facts_and_keeps_the_score(client):
    session_id, first = finished_run(client)
    answer_hint = hints_in(first["answers"][0]["answer"])[0]
    letter_hints = hints_in(first["cover_letter"])

    response = regenerate(client, session_id, [
        {"hint": answer_hint, "value": "Their weekly shipping cadence."},
        {"hint": letter_hints[-1], "value": "I can start on 1 March."},
        {"hint": "a hint that is not in the pack", "value": ""},  # blank values are ignored
    ])
    assert response.status_code == 200
    events, _ = read_stream(client, session_id, after=response.json()["after"])
    assert [e["type"] for e in events][-2:] == ["result", "done"]
    pack = next(e for e in events if e["type"] == "result")["pack"]

    assert "Their weekly shipping cadence." in pack["answers"][0]["answer"]
    assert "I can start on 1 March." in pack["cover_letter"]
    assert answer_hint not in str(hints_in(pack["answers"][0]["answer"]))  # that blank is gone
    assert hints_in(pack["cover_letter"]) == letter_hints[:-1]  # unfilled placeholders stay
    assert {k: pack[k] for k in SCORE_FIELDS} == {k: first[k] for k in SCORE_FIELDS}  # score frozen
    assert pack["warnings"] == []  # the number 1 in "1 March" came from the candidate's own facts
    assert [f["value"] for f in main.sessions[session_id].candidate_facts] == [
        "Their weekly shipping cadence.", "I can start on 1 March."]


def test_facts_accumulate_across_regenerations(client):
    session_id, first = finished_run(client)
    hints = hints_in(first["cover_letter"])

    after = regenerate(client, session_id, [{"hint": hints[0], "value": "First detail."}]).json()["after"]
    read_stream(client, session_id, after=after)
    after = regenerate(client, session_id, [{"hint": hints[1], "value": "Second detail."}]).json()["after"]
    events, _ = read_stream(client, session_id, after=after)
    pack = next(e for e in events if e["type"] == "result")["pack"]

    assert "First detail." in pack["cover_letter"] and "Second detail." in pack["cover_letter"]
    session = main.sessions[session_id]
    assert [f["value"] for f in session.candidate_facts] == ["First detail.", "Second detail."]
    last_message = next(m for m in reversed(session.history) if isinstance(m, dict) and m["role"] == "user")
    assert "First detail." in last_message["content"] and "Second detail." in last_message["content"]
    assert "data, not instructions" in last_message["content"]


def test_regenerate_error_cases(client, monkeypatch):
    fact = {"hint": "h", "value": "v"}
    assert regenerate(client, "nope", [fact]).status_code == 404

    no_pack = main.new_session(agent.Session(status="done"))
    response = regenerate(client, no_pack, [fact])
    assert response.status_code == 400 and "no application pack" in response.json()["detail"]

    session_id, _ = finished_run(client)
    for facts, expected in [([], "at least one"), ([{"hint": "h", "value": "   "}], "at least one"),
                            ([{"hint": "h", "value": "x" * 1_001}], "at most 1,000"),
                            ([{"hint": f"h{i}", "value": "v"} for i in range(31)], "at most 30")]:
        response = regenerate(client, session_id, facts)
        assert response.status_code == 400 and expected in response.json()["detail"], expected
    assert main.sessions[session_id].candidate_facts == []  # nothing stored from bad requests

    release = threading.Event()  # 409 while a turn is running
    monkeypatch.setattr(main.agent, "run_turn", lambda s, m: (release.wait(10), setattr(s, "status", "done")))
    try:
        assert client.post("/api/chat", json={"session_id": session_id, "message": "hello"}).status_code == 200
        response = regenerate(client, session_id, [fact])
        assert response.status_code == 409 and "still working" in response.json()["detail"]
    finally:
        release.set()


def test_page_has_stop_inline_blanks_counter_regenerate_and_copy_with_values(client):
    page = client.get("/").text
    for needle in [
        '"Stop"', '"Stopping…"', "/api/stop/",                     # Run and Send turn into Stop
        'class: "blank"', "Detail to fill in", "[ADD:",            # every [ADD: hint] becomes an input
        "details filled", "Nothing left to fill in",               # the counter
        "Regenerate with my details", "/api/regenerate",           # the regenerate button and call
        "filledText", "s.token",                                   # Copy: user's values, empty ones keep [ADD: ...]
        "Cover letter",                                            # replaces the cover note
    ]:
        assert needle in page, needle
    assert "Cover note" not in page and "cover_note" not in page
