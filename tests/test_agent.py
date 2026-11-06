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
    assert 250 <= len(pack["cover_letter"].split()) <= 350 and pack["cover_letter"].startswith("Dear Hiring Team,")
    assert "[ADD:" in pack["cover_letter"]
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
    pack = {**fake_llm.PACK, "cover_letter": fake_llm.PACK["cover_letter"] + " I am passionate about this."}
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


# --- Stop -----------------------------------------------------------------------------------

import threading  # noqa: E402
import time  # noqa: E402


def blocking_model(monkeypatch, response):
    """A model call that hangs until released, like a slow provider request."""
    entered, release = threading.Event(), threading.Event()

    def fake_chat(system, history, tools=None, on_wait=None, tool_choice=None):
        entered.set()
        release.wait(timeout=10)
        return response

    monkeypatch.setattr(agent.llm, "chat", fake_chat)
    return entered, release


def test_stop_before_the_first_model_call(monkeypatch):
    session = make_session()
    script(monkeypatch, [])  # any model call would fail the test
    session.stop_requested = True
    start(session)

    assert session.status == "stopped" and session.model_calls == 0
    assert types(session) == ["stopped"] and session.events[0]["message"] == "Stopped by you"
    assert session.history[-1] == {"role": "assistant", "content": "(Stopped by the user.)"}
    assert session.stop_requested is False  # cleared with the status


def test_stop_during_a_model_call_returns_at_once_and_discards_the_late_reply(monkeypatch):
    session = make_session()
    late = fake_llm.tool_reply("Late.", [("check_resume_for", {"skills": ["Python"]})])
    entered, release = blocking_model(monkeypatch, late)
    worker = threading.Thread(target=start, args=(session,))
    worker.start()
    try:
        assert entered.wait(5)
        assert session.request_stop() is True
        worker.join(3)
        assert not worker.is_alive()  # the turn ended although the model call is still hanging
        assert session.status == "stopped" and types(session) == ["stopped"]
        history_before, events_before = list(session.history), list(session.events)
    finally:
        release.set()  # the provider finally answers
    time.sleep(0.4)
    assert session.history == history_before and session.events == events_before  # reply discarded
    assert session.model_calls == 0 and session.pack is None


def test_stop_between_tool_calls_cancels_the_rest(monkeypatch):
    session = make_session()
    two_calls = fake_llm.tool_reply("Checking twice.", [("check_resume_for", {"skills": ["Python"]}),
                                                         ("check_resume_for", {"skills": ["SQL"]})])
    script(monkeypatch, [two_calls])
    real_run_tool = agent.tools.run_tool

    def run_then_stop(sess, name, args):
        result = real_run_tool(sess, name, args)
        sess.request_stop()  # the user presses Stop while the first tool runs
        return result

    monkeypatch.setattr(agent.tools, "run_tool", run_then_stop)
    start(session)

    assert session.status == "stopped" and types(session).count("tool_end") == 1
    tool_messages = [m for m in session.history if isinstance(m, dict) and m["role"] == "tool"]
    assert len(tool_messages) == 2 and "Cancelled by the user" in tool_messages[1]["content"]


def test_stopped_turn_keeps_the_pack_and_followups_still_work():
    session = make_session()
    start(session)
    pack = session.pack

    session.stop_requested = True
    agent.run_turn(session, "Please make answer 1 shorter.")
    assert session.status == "stopped" and session.pack is pack  # nothing lost

    agent.run_turn(session, "What is my biggest gap?")  # the Stop did not leak into this turn
    assert session.status == "done" and session.events[-1]["type"] == "reply"


def test_request_stop_only_works_while_running():
    session = agent.Session()
    assert session.request_stop() is False and session.stop_requested is False
    session.status = "running"
    assert session.request_stop() is True and session.stop_requested is True


# --- the score is frozen once a pack exists ------------------------------------------------------------

SCORE_FIELDS = ("score", "verdict", "missing_must_haves", "breakdown")


def test_regenerate_keeps_score_verdict_and_breakdown():
    session = make_session()
    start(session)
    before = {key: session.pack[key] for key in SCORE_FIELDS}

    facts = [{"hint": "your availability or start date, and how to reach you.", "value": "I can start in March."}]
    agent.run_turn(session, agent.build_facts_message(facts))

    assert session.status == "done" and "I can start in March." in session.pack["cover_letter"]
    assert {key: session.pack[key] for key in SCORE_FIELDS} == before


def test_model_cannot_rescore_in_a_follow_up_turn(monkeypatch):
    session = make_session()
    offered = []

    def run(replies):
        replies = iter(replies)

        def fake_chat(system, history, tools=None, on_wait=None, tool_choice=None):
            offered.append([t["function"]["name"] for t in tools])
            return next(replies)

        monkeypatch.setattr(agent.llm, "chat", fake_chat)

    run(good_run())
    start(session)
    assert "calculate_fit_score" in offered[0]  # the first run may score
    fit_before, pack_before = session.fit, {key: session.pack[key] for key in SCORE_FIELDS}

    # In a follow-up the model tries to re-score anyway (everything "missing"), then saves
    worse = [dict(r, match="missing") for r in fake_llm.REQUIREMENTS]
    offered.clear()
    run([fake_llm.tool_reply("Re-scoring.", [("calculate_fit_score", {"requirements": worse})]),
         fake_llm.tool_reply("Saving.", [("save_application_pack", fake_llm.PACK)])])
    agent.run_turn(session, agent.build_facts_message([{"hint": "h", "value": "v"}]))

    assert all("calculate_fit_score" not in names for names in offered)  # not even offered
    ends = [e for e in session.events_since(0) if e["type"] == "tool_end"]
    assert ends[-2]["ok"] is False and "fixed" in ends[-2]["summary"]  # and refused if called
    assert session.fit is fit_before
    assert {key: session.pack[key] for key in SCORE_FIELDS} == pack_before
