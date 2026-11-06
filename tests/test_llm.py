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


def test_fake_llm_honors_tool_choice(monkeypatch):
    monkeypatch.setenv("FAKE_LLM", "1")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)  # no key needed when fake
    history = [{"role": "user", "content": "=== JOB POST ==="}]
    forced = {"type": "function", "function": {"name": "save_application_pack"}}

    message = llm.append_reply(history, llm.chat("sys", history, [], tool_choice=forced))
    assert message.tool_calls[0].function.name == "save_application_pack"
    assert history[-1] is message  # appended unchanged, same object


def test_tool_choice_is_passed_to_litellm(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "ollama/x")
    monkeypatch.delenv("FAKE_LLM", raising=False)
    seen = {}
    monkeypatch.setattr(llm.litellm, "completion", lambda **kw: seen.update(kw) or "ok")
    tools = [{"type": "function", "function": {"name": "t", "parameters": {}}}]
    choice = {"type": "function", "function": {"name": "t"}}
    llm.chat("s", [], tools, tool_choice=choice)
    assert seen["tool_choice"] == choice


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


def _patch_completion(monkeypatch, error_factory):
    monkeypatch.setenv("LLM_MODEL", "ollama/x")
    monkeypatch.delenv("FAKE_LLM", raising=False)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    calls = []

    def always_fails(**kwargs):
        calls.append(1)
        raise error_factory()

    monkeypatch.setattr(llm.litellm, "completion", always_fails)
    return calls


def test_timeout_is_retried_only_once(monkeypatch):
    calls = _patch_completion(monkeypatch, lambda: llm.litellm.Timeout("slow", "x", "ollama"))
    with pytest.raises(llm.litellm.Timeout):
        llm.chat("s", [])
    assert len(calls) == 2  # first try + 1 retry


def test_connection_error_is_retried_only_once(monkeypatch):
    calls = _patch_completion(
        monkeypatch, lambda: llm.litellm.APIConnectionError("down", "ollama", "x"))
    with pytest.raises(llm.litellm.APIConnectionError):
        llm.chat("s", [])
    assert len(calls) == 2  # not 5, even though litellm reports status 500


def test_other_errors_are_not_retried(monkeypatch):
    calls = _patch_completion(monkeypatch, lambda: ValueError("bad request"))
    with pytest.raises(ValueError):
        llm.chat("s", [])
    assert len(calls) == 1


def test_litellm_tool_calls_have_their_hidden_dependency():
    # litellm imports this module on every completion() call that passes tools;
    # it needs the `mcp` package, which litellm does not declare (see requirements.txt)
    import importlib
    importlib.import_module("litellm.responses.mcp.chat_completions_handler")
