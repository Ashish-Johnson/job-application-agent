"""LiteLLM wrapper - the only file that knows about providers."""
import os
import re
import time

import litellm
from dotenv import load_dotenv
from litellm import Message, ModelResponse

from app import fake_llm

load_dotenv()

DEFAULT_MODEL = "gemini/gemini-3.5-flash"
MAX_RETRIES = 4  # rate limits and 5xx
MAX_RETRIES_SLOW = 1  # timeouts and connection errors
FIRST_WAIT = 5  # seconds; doubles on each retry
# Writing the full application pack on a thinking model can take over a minute
REQUEST_TIMEOUT = 120  # seconds per model call

# Provider prefix of LLM_MODEL -> env var that must hold its key.
# Ollama runs locally and needs none.
KEY_VARS = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


class LLMConfigError(Exception):
    """Bad or missing configuration; the message says what to fix."""


def get_model() -> str:
    return os.getenv("LLM_MODEL") or DEFAULT_MODEL


def fake_enabled() -> bool:
    return os.getenv("FAKE_LLM") == "1"


def check_config() -> str:
    """Return the model name, or raise LLMConfigError if its key is missing."""
    model = get_model()
    if fake_enabled():
        return "fake"
    key_var = KEY_VARS.get(model.split("/")[0])
    if key_var and not os.getenv(key_var):
        raise LLMConfigError(
            f"LLM_MODEL is '{model}' but {key_var} is not set. "
            f"Add {key_var}=... to your .env file (see .env.example)."
        )
    return model


def _retry_delay(err: Exception, attempt: int) -> float:
    """Provider's retry-after if it gave one, else exponential backoff."""
    headers = getattr(getattr(err, "response", None), "headers", None) or {}
    if headers.get("retry-after"):
        try:
            return float(headers["retry-after"])
        except ValueError:
            pass
    # Gemini puts the delay in the message body, e.g. "Please retry in 12.5s"
    match = re.search(r"retry in ([\d.]+)s", str(err), re.IGNORECASE)
    if match:
        return float(match.group(1))
    return FIRST_WAIT * 2**attempt


def _max_retries(err: Exception) -> int:
    """How many retries this error deserves (0 = fail immediately)."""
    # Checked first: litellm gives APIConnectionError a 500 status code, but a
    # dead connection or a very slow call is unlikely to fix itself quickly,
    # and each attempt can cost up to REQUEST_TIMEOUT seconds.
    if isinstance(err, (litellm.Timeout, litellm.APIConnectionError)):
        return MAX_RETRIES_SLOW
    status = getattr(err, "status_code", None)
    if isinstance(err, litellm.RateLimitError) or status == 429 or (
        isinstance(status, int) and status >= 500
    ):
        return MAX_RETRIES
    return 0


def chat(system: str, history: list, tools: list | None = None, on_wait=None,
         tool_choice: dict | None = None) -> ModelResponse:
    """One model call: system prompt + history + tool schemas -> response.

    Retries rate limits (429) and 5xx errors up to 4 times, timeouts and
    connection errors once, calling on_wait(message) before each sleep so the
    UI can show it. tool_choice (OpenAI format) can force a specific tool.
    """
    model = check_config()
    if fake_enabled():
        return fake_llm.respond(history, tools, tool_choice)

    messages = [{"role": "system", "content": system}] + history
    attempt = 0
    while True:
        try:
            # tools are passed through; we run the tool loop ourselves
            # timeout so a hung request fails (and retries) instead of freezing the UI
            extra = {"tool_choice": tool_choice} if tools and tool_choice else {}
            return litellm.completion(model=model, messages=messages, tools=tools or None,
                                      timeout=REQUEST_TIMEOUT, **extra)
        except Exception as err:
            limit = _max_retries(err)
            if attempt >= limit:
                raise
            delay = _retry_delay(err, attempt)
            attempt += 1
            if on_wait:
                on_wait(f"Rate limited or provider busy, retrying in {delay:.0f}s "
                        f"(retry {attempt} of {limit})")
            time.sleep(delay)


def append_reply(history: list, response: ModelResponse) -> Message:
    """Append the model's message to history exactly as returned.

    Do not rebuild it from dict fields: provider_specific_fields (Gemini's
    thought signatures) must be sent back untouched or later calls can fail.
    """
    message = response.choices[0].message
    history.append(message)
    return message
