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

PACK = dict(role="r", company="c", summary="s", strengths=[], gaps=[], answers=[], cover_letter="n")


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


# --- honesty check ----------------------------------------------------------------

import copy  # noqa: E402

from app import fake_llm  # noqa: E402

JOB = (Path(__file__).parent.parent / "samples" / "job_post.txt").read_text(encoding="utf-8")


@pytest.fixture
def scored():
    s = Session(resume_text=RESUME, job_text=JOB)
    tools.calculate_fit_score(s, [req("must", "strong")])
    return s


def good_pack():
    return copy.deepcopy(fake_llm.PACK)


def save(session, pack):
    return tools.save_application_pack(session, **pack)


def test_clean_pack_passes_the_honesty_check(scored):
    result = save(scored, good_pack())
    assert result["saved"] and scored.pack["warnings"] == []
    assert scored.honesty_rejections == 0


def test_invented_number_is_rejected(scored):
    pack = good_pack()
    pack["answers"][0]["answer"] += " I also cut hosting costs by 37%."
    result = save(scored, pack)
    assert "error" in result and scored.pack is None
    assert any('"37"' in issue and "answer 1" in issue for issue in result["issues"])
    assert "37%" in result["issues"][0]  # shows the offending text


def test_numbers_from_resume_or_job_post_are_fine(scored):
    pack = good_pack()
    # 40 endpoints, 9s and 30% style numbers: 40 and 9 are in the resume, 25 in the job post
    pack["answers"][0]["answer"] += " I built around 40 endpoints, cut a 9s call, and know a team of 25."
    assert save(scored, pack)["saved"]


def test_numbers_inside_placeholders_are_ignored(scored):
    pack = good_pack()
    pack["cover_letter"] += " [ADD: your 2024 result of 12%]"
    assert save(scored, pack)["saved"]


def test_numbers_in_gap_advice_are_allowed_but_not_in_the_gap_itself(scored):
    pack = good_pack()
    pack["gaps"][0]["how_to_address"] = "Spend 6 weeks on an evaluation course, 3 evenings a week."
    assert save(scored, pack)["saved"]  # advice, not a claim about the candidate

    scored.pack = None
    pack["gaps"][0]["gap"] = "Only 6 months of evaluation experience."  # a claim: checked
    assert any("gap 1 (gap)" in i and '"6"' in i for i in save(scored, pack)["issues"])


def test_feelings_in_gap_advice_are_still_rejected(scored):
    pack = good_pack()
    pack["gaps"][0]["how_to_address"] = "Say you are passionate about evaluation."
    assert any("how_to_address" in i and '"passionate"' in i for i in save(scored, pack)["issues"])


def test_fake_evidence_is_rejected(scored):
    pack = good_pack()
    pack["strengths"][0]["evidence"] = "Led a team of ten engineers at a bank."
    result = save(scored, pack)
    assert "error" in result and "Strength 1" in result["issues"][0]


def test_evidence_matching_ignores_case_whitespace_and_ellipsis(scored):
    pack = good_pack()
    pack["strengths"][0]["evidence"] = '- "build and   MAINTAIN internal rest apis ... used by the dispatch team"'
    assert save(scored, pack)["saved"]


def test_feeling_phrase_is_rejected(scored):
    pack = good_pack()
    pack["cover_letter"] = "I am excited about Corvane Pay and passionate about agents."
    issues = save(scored, pack)["issues"]
    assert any('"excited"' in i for i in issues) and any('"passionate"' in i for i in issues)


def test_feeling_inside_placeholder_is_fine_and_resume_stated_feelings_too(scored):
    pack = good_pack()
    pack["cover_letter"] += " [ADD: why you are excited about this team]"
    assert save(scored, pack)["saved"]

    scored.pack = None
    scored.resume_text += "\nI am passionate about teaching junior developers."
    pack["answers"][0]["answer"] += " I am passionate about teaching."
    assert save(scored, pack)["saved"]


def test_after_two_rejections_it_saves_with_warnings(scored):
    bad = good_pack()
    bad["answers"][0]["answer"] += " I raised revenue by 37%."
    assert "error" in save(scored, bad) and "error" in save(scored, bad)
    assert scored.honesty_rejections == 2 and scored.pack is None

    result = save(scored, bad)  # third attempt: saved anyway
    assert result["saved"] and result["warnings"] == 1
    assert len(scored.pack["warnings"]) == 1 and "37" in scored.pack["warnings"][0]
    assert tools.summary_for("save_application_pack", result) == "Application pack saved with 1 warning(s)"


def test_rejection_summary_text(scored):
    pack = good_pack()
    pack["cover_letter"] = "I love it, I am excited, and I am passionate, with 77 years of experience."
    result = save(scored, pack)
    assert tools.summary_for("save_application_pack", result) == (
        f"Honesty check: {len(result['issues'])} issues, asking the agent to fix them")


# --- the score is frozen, schemas follow ---------------------------------------------------------------

def test_calculate_fit_score_refuses_once_a_pack_exists(scored):
    save(scored, good_pack())
    fit_before = scored.fit
    result = tools.calculate_fit_score(scored, [req("must", "missing")])
    assert "error" in result and "fixed" in result["error"]
    assert scored.fit is fit_before and scored.pack["score"] == fit_before["score"]


def test_schemas_drop_calculate_fit_score_after_the_pack(scored):
    names = lambda s: [t["function"]["name"] for t in tools.schemas_for(s)]  # noqa: E731
    assert "calculate_fit_score" in names(scored)
    save(scored, good_pack())
    assert "calculate_fit_score" not in names(scored) and "save_application_pack" in names(scored)


# --- candidate facts count as true sources --------------------------------------------------------------

def test_number_only_in_candidate_facts_is_accepted(scored):
    pack = good_pack()
    pack["answers"][0]["answer"] += " I raised throughput to 450 requests per second."
    assert "450" in save(scored, pack)["issues"][0]  # rejected: not in resume or job post

    scored.candidate_facts = [{"hint": "your result", "value": "We reached 450 requests per second."}]
    assert save(scored, pack)["saved"] and scored.pack["warnings"] == []


def test_feeling_is_accepted_via_a_fact_value_but_never_via_a_hint(scored):
    pack = good_pack()
    pack["cover_letter"] += " I am excited about payments."

    scored.candidate_facts = [{"hint": "why you are excited", "value": "Payments interest me."}]
    issues = save(scored, pack)["issues"]
    assert any('"excited"' in i and "cover letter" in i for i in issues)  # the hint does not count

    scored.honesty_rejections = 0
    scored.candidate_facts = [{"hint": "motivation", "value": "I am excited about payments."}]
    assert save(scored, pack)["saved"]
