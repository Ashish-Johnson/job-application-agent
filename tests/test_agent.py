import itertools
from pathlib import Path

import pytest

from app import agent, fake_llm

SAMPLES = Path(__file__).parent.parent / "samples"
RESUME = (SAMPLES / "resume.txt").read_text(encoding="utf-8")
QUESTIONS = [q for q in (SAMPLES / "questions.txt").read_text(encoding="utf-8").splitlines() if q.strip()]


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    monkeypatch.setenv("FAKE_LLM", "1")


def make_session():
    return agent.Session(job_text=(SAMPLES / "job_post.txt").read_text(encoding="utf-8"),
                         resume_text=RESUME, questions=list(QUESTIONS))


def start(session):
    agent.run_turn(session, agent.build_task_message(session))


def types(session):
    return [e["type"] for e in session.events]


def script(monkeypatch, replies):
    """Replace llm.chat with a scripted sequence. Items are ModelResponses, exceptions,
    or functions of the tool_choice. Returns the list of tool_choices the loop sent."""
    choices = []
    replies = iter(replies)

    def fake_chat(system, history, tools=None, on_wait=None, tool_choice=None):
        choices.append(tool_choice["function"]["name"] if tool_choice else None)
        item = next(replies)
        if isinstance(item, Exception):
            raise item
        return item(tool_choice) if callable(item) else item

    monkeypatch.setattr(agent.llm, "chat", fake_chat)
    return choices


def good_run():
    """The three tool replies of a normal run."""
    return [fake_llm.tool_reply(fake_llm.THOUGHTS[n], [(n, fake_llm.ARGS[n])]) for n in fake_llm.ORDER]


def obey(tool_choice):
    """A reply that does whatever the loop forces."""
    name = tool_choice["function"]["name"]
    return fake_llm.tool_reply("Forced.", [(name, fake_llm.ARGS[name])])


def ignore(tool_choice):
    return fake_llm.tool_reply("Still checking.", [("check_resume_for", {"skills": ["Python"]})])


def auto_messages(session):
    return [m["content"] for m in session.history
            if isinstance(m, dict) and m["role"] == "user" and m["content"].startswith(agent.AUTO_PREFIX)]


# --- happy path ----------------------------------------------------------------

def test_happy_path():
    session = make_session()
    start(session)

    assert session.status == "done" and session.model_calls == 3
    assert types(session) == ["thought", "tool_start", "tool_end"] * 3 + ["result"]
    assert all(e["ok"] for e in session.events if e["type"] == "tool_end")

    ts = [e["t"] for e in session.events]
    assert ts == sorted(ts) and [e["id"] for e in session.events] == list(range(len(ts)))

    pack = session.pack
    assert pack["score"] == session.fit["score"] and pack["verdict"] == session.fit["verdict"]
    assert [a["question"] for a in pack["answers"]] == QUESTIONS
    assert all(60 <= len(a["answer"].split()) <= 120 for a in pack["answers"])
    assert len(pack["cover_note"].split()) < 120
    assert "[ADD:" in str(pack)  # missing facts are placeholders, not inventions
    # honesty: every strength's evidence is real resume text
    assert all(s["evidence"] in RESUME for s in pack["strengths"])


def test_wait_callback_becomes_event(monkeypatch):
    session = make_session()

    def chat_with_wait(system, history, tools=None, on_wait=None, tool_choice=None):
        on_wait("Rate limited, retrying in 5s")
        return fake_llm.respond(history, tools, tool_choice)

    monkeypatch.setattr(agent.llm, "chat", chat_with_wait)
    start(session)
    assert session.status == "done"
    assert [e["message"] for e in session.events if e["type"] == "wait"][0] == "Rate limited, retrying in 5s"


# --- guardrails ----------------------------------------------------------------

def test_nudge_once_when_model_stops_early(monkeypatch):
    session = make_session()
    script(monkeypatch, [fake_llm.text_reply("This looks like a strong match.")] + good_run())
    start(session)

    assert session.status == "done" and session.pack is not None
    nudges = auto_messages(session)
    assert len(nudges) == 1 and "save_application_pack" in nudges[0]


def test_forced_score_then_save_when_model_will_not_use_tools(monkeypatch):
    session = make_session()
    choices = script(monkeypatch, [fake_llm.text_reply("Hmm."), fake_llm.text_reply("Still thinking."),
                                   obey, obey])
    start(session)

    assert choices == [None, None, "calculate_fit_score", "save_application_pack"]
    assert session.status == "done" and session.pack["score"] == session.fit["score"]


def test_forced_on_last_two_steps_when_model_keeps_looping(monkeypatch):
    session = make_session()
    choices = script(monkeypatch, [ignore] * 10 + [obey, obey])
    start(session)

    assert choices == [None] * 10 + ["calculate_fit_score", "save_application_pack"]
    assert session.status == "done" and session.model_calls == agent.MAX_STEPS


def test_model_that_ignores_forcing_ends_with_error_event(monkeypatch):
    session = make_session()
    script(monkeypatch, itertools.repeat(ignore))
    start(session)

    assert session.status == "error" and session.pack is None
    assert types(session)[-1] == "error"
    assert "without finishing" in session.events[-1]["message"]


def test_need_input_is_a_clear_error(monkeypatch):
    session = make_session()
    script(monkeypatch, [fake_llm.text_reply("NEED_INPUT: paste the job post text.")])
    start(session)
    assert session.status == "error" and session.events[-1]["message"] == "paste the job post text."


# --- empty replies ---------------------------------------------------------------

def test_empty_reply_gets_a_continue_message(monkeypatch):
    session = make_session()
    script(monkeypatch, [fake_llm.text_reply(""), fake_llm.text_reply("")] + good_run())
    start(session)

    assert session.status == "done"
    assert auto_messages(session) == [agent.CONTINUE, agent.CONTINUE]


def test_too_many_empty_replies_is_an_error(monkeypatch):
    session = make_session()
    script(monkeypatch, itertools.repeat(fake_llm.text_reply("")))
    start(session)
    assert session.status == "error" and "empty" in session.events[-1]["message"]
    assert session.model_calls == agent.MAX_EMPTY_REPLIES + 1


# --- tool errors don't crash the loop -------------------------------------------------

def test_bad_tool_calls_do_not_crash_the_loop(monkeypatch):
    session = make_session()
    bad = [
        fake_llm.tool_reply("Checking.", [("check_resume_for", {})]),          # missing argument
        fake_llm.tool_reply("Checking again.", [("check_resume_for", "{oops")]),  # not JSON
        fake_llm.tool_reply("Trying.", [("no_such_tool", {})]),                # unknown tool
    ]
    script(monkeypatch, bad + good_run())
    start(session)

    ends = [e for e in session.events if e["type"] == "tool_end"]
    assert [e["ok"] for e in ends] == [False, False, False, True, True, True]
    assert all(e["summary"].startswith("Failed:") for e in ends[:3])
    assert session.status == "done" and session.pack is not None
    # every tool call got a matching tool message
    tool_msgs = [m for m in session.history if isinstance(m, dict) and m["role"] == "tool"]
    assert len(tool_msgs) == 6


def test_exception_in_model_call_ends_with_error(monkeypatch):
    session = make_session()
    script(monkeypatch, [RuntimeError("boom")])
    start(session)
    assert session.status == "error" and "boom" in session.events[-1]["message"]


def test_missing_api_key_is_reported(monkeypatch):
    monkeypatch.delenv("FAKE_LLM")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-4o-mini")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    session = make_session()
    start(session)
    assert session.status == "error" and "OPENAI_API_KEY" in session.events[-1]["message"]


# --- follow-up turns -----------------------------------------------------------------

def test_follow_up_change_resaves_the_full_pack():
    session = make_session()
    start(session)
    before = session.pack
    seen = len(session.events)

    agent.run_turn(session, "Please make answer 2 shorter.")

    after = session.pack
    assert session.status == "done" and types_since(session, seen)[-1] == "result"
    assert len(after["answers"][1]["answer"].split()) < len(before["answers"][1]["answer"].split())
    assert after["answers"][0] == before["answers"][0] and after["strengths"] == before["strengths"]
    assert after["score"] == before["score"]
    assert session.events[seen]["t"] < 5  # the clock restarts each turn


def test_follow_up_question_gets_a_reply_and_keeps_the_pack():
    session = make_session()
    start(session)
    before, seen = session.pack, len(session.events)

    agent.run_turn(session, "What is my biggest gap?")

    assert session.status == "done" and session.pack is before
    assert types_since(session, seen) == ["reply"]


def types_since(session, index):
    return [e["type"] for e in session.events_since(index)]


# --- honesty check inside the loop ----------------------------------------------------

def bad_save(session_pack=None):
    pack = {**fake_llm.PACK, "cover_note": fake_llm.PACK["cover_note"] + " I am passionate about this."}
    return fake_llm.tool_reply("Saving.", [("save_application_pack", pack)])


def test_rejected_pack_is_fixed_and_resaved(monkeypatch):
    session = make_session()
    run = good_run()
    script(monkeypatch, run[:2] + [bad_save(), run[2]])
    start(session)

    ends = [e for e in session.events if e["type"] == "tool_end"]
    assert [e["ok"] for e in ends] == [True, True, False, True]
    assert ends[2]["summary"] == "Honesty check: 1 issue, asking the agent to fix it"
    assert "passionate" in ends[2]["issues"][0]
    assert session.status == "done" and session.pack["warnings"] == []


def test_model_that_never_fixes_the_pack_still_ends_with_a_saved_pack(monkeypatch):
    session = make_session()
    script(monkeypatch, good_run()[:2] + [bad_save(), bad_save(), bad_save()])
    start(session)

    assert session.status == "done" and session.model_calls == 5
    assert len(session.pack["warnings"]) == 1 and "passionate" in session.pack["warnings"][0]
    assert session.events[-1]["type"] == "result"


def test_forced_final_save_never_gets_rejected(monkeypatch):
    session = make_session()
    # the model loops until the loop forces score then save; its forced save is "bad"
    def forced_bad_save(tool_choice):
        return bad_save() if tool_choice["function"]["name"] == "save_application_pack" else obey(tool_choice)
    script(monkeypatch, [ignore] * 10 + [obey, forced_bad_save])
    start(session)

    assert session.status == "done" and len(session.pack["warnings"]) == 1


def test_rejection_counter_resets_each_turn(monkeypatch):
    session = make_session()
    script(monkeypatch, good_run()[:2] + [bad_save(), good_run()[2]])
    start(session)
    session.honesty_rejections = 2  # as if the last turn used its attempts
    script(monkeypatch, [good_run()[2]])
    agent.run_turn(session, "Please save it again.")
    assert session.honesty_rejections == 0
