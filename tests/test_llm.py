import pytest

from app import llm


def test_missing_key_names_the_variable(monkeypatch):
    monkeypatch.delenv("FAKE_LLM", raising=False)
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-4o-mini")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(llm.LLMConfigError, match="OPENAI_API_KEY"):
        llm.check_config()


def test_ollama_needs_no_key(monkeypatch):
    monkeypatch.delenv("FAKE_LLM", raising=False)
    monkeypatch.setenv("LLM_MODEL", "ollama/qwen3:8b")
    assert llm.check_config() == "ollama/qwen3:8b"


def test_fake_llm_calls_tools_then_answers(monkeypatch):
    monkeypatch.setenv("FAKE_LLM", "1")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)  # no key needed when fake
    tools = [{"type": "function", "function": {"name": "tool_a", "parameters": {}}}]
    history = [{"role": "user", "content": "go"}]

    first = llm.append_reply(history, llm.chat("sys", history, tools))
    assert first.tool_calls[0].function.name == "tool_a"
    assert history[-1] is first  # appended unchanged, same object

    history.append({"role": "tool", "tool_call_id": "fake_0", "content": "ok"})
    last = llm.append_reply(history, llm.chat("sys", history, tools))
    assert last.content == "FAKE: done." and not last.tool_calls


def test_retries_rate_limit_then_succeeds(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "ollama/x")
    monkeypatch.delenv("FAKE_LLM", raising=False)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    calls = []

    def flaky(**kwargs):
        calls.append(1)
        if len(calls) < 3:
            raise llm.litellm.RateLimitError("slow down", "ollama", "x")
        return "ok"

    monkeypatch.setattr(llm.litellm, "completion", flaky)
    waits = []
    assert llm.chat("s", [], on_wait=waits.append) == "ok"
    assert len(calls) == 3 and len(waits) == 2
