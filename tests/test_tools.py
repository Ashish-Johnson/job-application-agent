from pathlib import Path

import pytest

from app import tools
from app.agent import Session

RESUME = (Path(__file__).parent.parent / "samples" / "resume.txt").read_text(encoding="utf-8")


@pytest.fixture
def session():
    return Session(resume_text=RESUME)


# --- check_resume_for ---------------------------------------------------------

def by_skill(result):
    return {r["skill"]: r for r in result["results"]}


def test_resume_finds_skills_case_insensitive(session):
    found = by_skill(tools.check_resume_for(session, ["python", "FASTAPI", "PostgreSQL"]))
    assert all(r["found"] for r in found.values())
    assert any("FastAPI" in line for line in found["FASTAPI"]["evidence"])


def test_resume_punctuation_tolerant(session):
    session.resume_text = "Built apps with Node.js and Next JS. Also CI-CD."
    found = by_skill(tools.check_resume_for(session, ["nodejs", "Next.js", "ci/cd"]))
    assert found["nodejs"]["found"] and found["Next.js"]["found"] and found["ci/cd"]["found"]


def test_resume_alternatives_and_missing(session):
    found = by_skill(tools.check_resume_for(session, ["Flask/FastAPI", "Kubernetes", "evals"]))
    assert found["Flask/FastAPI"]["found"]
    assert not found["Kubernetes"]["found"] and found["Kubernetes"]["evidence"] == []
    assert not found["evals"]["found"]


def test_resume_no_partial_word_matches(session):
    session.resume_text = "Wrote JavaScript at Google using C++."
    found = by_skill(tools.check_resume_for(session, ["Java", "Go", "C", "C++"]))
    assert not found["Java"]["found"] and not found["Go"]["found"] and not found["C"]["found"]
    assert found["C++"]["found"]


def test_resume_evidence_capped_at_two_lines(session):
    session.resume_text = "Python one\nPython two\nPython three"
    assert len(by_skill(tools.check_resume_for(session, ["Python"]))["Python"]["evidence"]) == 2


def test_resume_empty_input_is_an_error(session):
    assert "error" in tools.check_resume_for(session, [])


# --- calculate_fit_score ------------------------------------------------------

def req(importance, match, name="x"):
    return {"requirement": name, "importance": importance, "match": match}


def test_score_math(session):
    # must strong (2/2) + must partial (1/2) + nice missing (0/1) = 3/5 = 60
    result = tools.calculate_fit_score(
        session, [req("must", "strong"), req("must", "partial"), req("nice", "missing")])
    assert result["score"] == 60
    assert result["verdict"] == "Good fit with gaps"
    assert session.fit is result


def test_strong_fit_needs_no_missing_must(session):
    good = [req("must", "strong")] * 3 + [req("nice", "missing")]
    assert tools.calculate_fit_score(session, good)["verdict"] == "Strong fit"  # 6/7 = 86

    # score 80 but a must-have is missing: not "Strong fit"
    reqs = [req("must", "missing", "Kubernetes")] + [req("nice", "strong")] * 8
    result = tools.calculate_fit_score(session, reqs)
    assert result["score"] == 80 and result["missing_must_haves"] == ["Kubernetes"]
    assert result["verdict"] == "Good fit with gaps"


def test_stretch_role(session):
    result = tools.calculate_fit_score(
        session, [req("must", "missing"), req("must", "partial"), req("nice", "missing")])
    assert result["score"] == 20 and result["verdict"] == "Stretch role"


def test_score_rejects_bad_entries(session):
    assert "error" in tools.calculate_fit_score(session, [req("must", "great")])
    assert "error" in tools.calculate_fit_score(session, [])
    assert session.fit is None


# --- save_application_pack ----------------------------------------------------

PACK = dict(role="r", company="c", summary="s", strengths=[], gaps=[], answers=[], cover_note="n")


def test_pack_requires_a_score_first(session):
    assert "error" in tools.save_application_pack(session, **PACK)
    assert session.pack is None


def test_pack_always_uses_calculated_score(session):
    tools.calculate_fit_score(session, [req("must", "strong"), req("nice", "missing")])
    tools.save_application_pack(session, **PACK)
    assert session.pack["score"] == 67 and session.pack["verdict"] == "Good fit with gaps"
    # a model-supplied score is not even an accepted argument
    result = tools.run_tool(session, "save_application_pack", {**PACK, "score": 99})
    assert "error" in result and session.pack["score"] == 67


# --- URL safety (no network: IP literals and localhost need no DNS) ----------

@pytest.mark.parametrize("url", [
    "http://localhost:8000/admin",
    "http://127.0.0.1/",
    "http://10.1.2.3/",
    "http://192.168.0.10/",
    "http://172.16.5.5/",
    "http://169.254.169.254/latest/meta-data",
    "http://0.0.0.0/",
    "http://[::1]/",
    "ftp://example.com/file",
    "file:///etc/passwd",
    "not a url",
])
def test_unsafe_urls_are_refused(url):
    assert tools.check_public_url(url) is not None


def test_public_ip_is_allowed():
    assert tools.check_public_url("https://93.184.216.34/jobs") is None


def test_fetch_refuses_private_address_without_network(session):
    result = tools.fetch_webpage(session, "http://127.0.0.1:8000/")
    assert "error" in result


# --- dispatcher and UI text ---------------------------------------------------

def test_run_tool_never_raises(session):
    assert "error" in tools.run_tool(session, "nope", {})
    assert "error" in tools.run_tool(session, "check_resume_for", {})  # missing argument


def test_ui_labels_and_summaries(session):
    args = {"skills": ["Python", "Kubernetes", "evals"]}
    assert tools.label_for("check_resume_for", args) == "Checking your resume for 3 skills"
    result = tools.run_tool(session, "check_resume_for", args)
    assert tools.summary_for("check_resume_for", result) == "Found 1 of 3 · missing: Kubernetes, evals"


def test_every_schema_has_a_function():
    names = [s["function"]["name"] for s in tools.SCHEMAS]
    assert sorted(names) == sorted(tools.FUNCTIONS) and len(names) == 5


def test_user_agent_carries_contact(monkeypatch):
    monkeypatch.setenv("CONTACT_URL", "https://example.org/me")
    assert "https://example.org/me" in tools.user_agent()


def test_pack_strips_copied_question_numbers(session):
    tools.calculate_fit_score(session, [req("must", "strong")])
    answers = [{"question": "1. Why us?", "answer": "a"}, {"question": "2) Why you?", "answer": "b"},
               {"question": "Is 3 a crowd?", "answer": "c"}]
    tools.save_application_pack(session, **{**PACK, "answers": answers})
    assert [a["question"] for a in session.pack["answers"]] == ["Why us?", "Why you?", "Is 3 a crowd?"]
