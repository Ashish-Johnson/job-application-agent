"""Fake model for FAKE_LLM=1: a believable, scripted run on the samples/ files.

No network, no quota. It decides its next move from what is already in the
history, so it works for the first run, for forced tool_choice, and for simple
follow-ups ("make answer 2 shorter"). Tests also use text_reply/tool_reply to
script exact sequences.
"""
import itertools
import json
import os
import re
import time

from litellm import Message, ModelResponse
from litellm.types.utils import ChatCompletionMessageToolCall, Function

# Prefix of messages the agent loop writes itself (nudges, "continue").
# Must match AUTO_PREFIX in agent.py (agent.py can't be imported from here).
AUTO_PREFIX = "[auto] "
# Must match FACTS_PREFIX in agent.py: the start of the "Regenerate with my details" message.
FACTS_PREFIX = "Here are true details provided by the candidate"

_ids = itertools.count(1)

# --- scripted run on samples/ -------------------------------------------------

SKILLS = ["Python", "FastAPI/Flask", "OpenAI/Gemini/LLM", "function calling", "pytest",
          "PostgreSQL/SQL", "Docker", "GitHub Actions/CI", "agent", "evals/evaluation",
          "fintech/payments", "tracing/observability"]

REQUIREMENTS = [
    {"requirement": "1+ years of professional software development", "importance": "must", "match": "strong"},
    {"requirement": "Strong Python and HTTP APIs (FastAPI)", "importance": "must", "match": "strong"},
    {"requirement": "LLM APIs with tool/function calling", "importance": "must", "match": "strong"},
    {"requirement": "Automated tests", "importance": "must", "match": "strong"},
    {"requirement": "Clear written English", "importance": "must", "match": "partial"},
    {"requirement": "Agent loops or multi-step LLM workflows", "importance": "nice", "match": "partial"},
    {"requirement": "Evaluating LLM outputs", "importance": "nice", "match": "missing"},
    {"requirement": "Fintech or payments background", "importance": "nice", "match": "missing"},
    {"requirement": "PostgreSQL and SQL", "importance": "nice", "match": "strong"},
    {"requirement": "Docker and CI/CD", "importance": "nice", "match": "partial"},
    {"requirement": "Observability or tracing for LLM apps", "importance": "nice", "match": "missing"},
]

# Evidence strings are exact fragments of samples/resume.txt (a test checks this).
PACK = {
    "role": "AI Builder (Agents)",
    "company": "Corvane Pay",
    "summary": "A solid match on Python, API building and hands-on LLM tool calling, "
               "with gaps in LLM evaluation, tracing and fintech experience.",
    "strengths": [
        {"strength": "Hands-on LLM work with tool/function calling",
         "evidence": "Uses function calling to look up pantry items from a JSON file."},
        {"strength": "Builds and maintains Python APIs used by real teams",
         "evidence": "Build and maintain internal REST APIs in Python (FastAPI) used by the dispatch team"},
        {"strength": "Writes automated tests and runs them in CI",
         "evidence": "Added pytest coverage for the booking service and set up a GitHub Actions pipeline"},
    ],
    "gaps": [
        {"gap": "No experience evaluating LLM output quality",
         "how_to_address": "Build a small test set of real delivery emails with the expected bookings, "
                           "score the parser against it, and mention the result [ADD: accuracy you measure]."},
        {"gap": "No fintech or payments background",
         "how_to_address": "Say so plainly, point to the structured-data work you did in logistics, "
                           "and read a short primer on payout reconciliation before any interview."},
        {"gap": "No tracing or monitoring of LLM calls",
         "how_to_address": "Add request logging with timings and token counts to Recipe Pantry Helper "
                           "and describe what you learned from the logs."},
    ],
    "answers": [
        {"question": "Why do you want to join Corvane Pay?",
         "answer": "Corvane Pay's job post describes agents that take over payout matching and "
                   "support drafts, with engineers working directly beside the operations team. "
                   "That is the work I want to do. At Tallowfield Logistics I already used an LLM "
                   "to turn messy delivery emails into structured bookings, which cut manual data "
                   "entry for the dispatch team by roughly half. I would like to do that with real "
                   "accountability around financial data. [ADD: one specific thing about Corvane "
                   "Pay's product or team that drew you in, beyond the job post.]"},
        {"question": "Tell us about something you built with an LLM and what you learned from it.",
         "answer": "At Tallowfield Logistics I wrote a Python service that sends customer delivery "
                   "emails to an OpenAI model and turns the reply into structured booking records. "
                   "It cut manual data entry for the dispatch team by roughly half. I learned to "
                   "treat model output as untrusted input: I parse it, check every field, and "
                   "send anything doubtful to a person. I also built Recipe Pantry Helper with "
                   "Gemini function calling, which about 30 people use. [ADD: one concrete mistake "
                   "the model made and how you fixed it.]"},
        {"question": "Describe a time you were stuck with no clear owner or spec. What did you do?",
         "answer": "The closest example on my resume is the reporting endpoint at Tallowfield "
                   "Logistics that took about 9 seconds to respond. [ADD: what made it unowned or "
                   "unspecified, and who you checked with.] I traced the slow queries, added indexes "
                   "and removed N+1 queries, which brought it under 2 seconds. [ADD: how you "
                   "confirmed that this was what the team actually needed, and what you would do "
                   "earlier next time.]"},
    ],
    "cover_letter": (
        'Dear Hiring Team,\n'
        '\n'
        "I am applying for the AI Builder (Agents) role at Corvane Pay. Your post describes agents that take over repetitive operations work, such as matching payouts to invoices and drafting replies to merchant support tickets, and that is close to work I have already done. [ADD: one specific thing about Corvane Pay's product or team that made you apply.]\n"
        '\n'
        'I have about two years of professional experience building back ends in Python. At Tallowfield Logistics I build and maintain internal REST APIs in FastAPI, around 40 endpoints backed by PostgreSQL, used by the dispatch team. I also wrote a Python service that parses customer delivery emails with an OpenAI model and turns them into structured booking records, which cut manual data entry for the dispatch team by roughly half. Outside work I built Recipe Pantry Helper, a small web app that uses Gemini function calling, and about 30 people use it regularly. I write pytest tests for the booking service and run them in GitHub Actions on every pull request.\n'
        '\n'
        'I want to be straightforward about what I have not done yet. I have not written evaluations for LLM output, I have not worked in payments or fintech, and I have not built a full multi-step agent loop beyond function calling. [ADD: what you have already started to do about these gaps, such as a test set or a side project.] Your post says you expect people to find the problem and ship a fix, and I have done that with a slow reporting endpoint that I brought from about 9s to under 2s by adding indexes and removing N+1 queries. [ADD: how you found that problem and who you worked with.]\n'
        '\n'
        'Thank you for considering my application. [ADD: your availability or start date, and how to reach you.]\n'
        '\n'
        'Kind regards,\n'
        'Maya Okafor-Lindqvist'),
}

ARGS = {
    "check_resume_for": {"skills": SKILLS},
    "calculate_fit_score": {"requirements": REQUIREMENTS},
    "save_application_pack": PACK,
}

THOUGHTS = {
    "check_resume_for": "I'll check your resume for the skills this job asks for, so every rating is backed by evidence.",
    "calculate_fit_score": "Now I'll rate each requirement from that evidence and calculate the fit score.",
    "save_application_pack": "I have what I need, so I'll write your strengths, gaps, answers and cover note and save the pack.",
}

ORDER = ["check_resume_for", "calculate_fit_score", "save_application_pack"]


# --- reply builders (also used by tests) --------------------------------------

def _response(message: Message, finish: str) -> ModelResponse:
    return ModelResponse(model="fake",
                         choices=[{"index": 0, "message": message, "finish_reason": finish}])


def text_reply(text: str) -> ModelResponse:
    return _response(Message(role="assistant", content=text), "stop")


def tool_reply(thought: str, calls: list) -> ModelResponse:
    """calls: [(tool_name, args)]; args is a dict, or a raw string to send as-is."""
    tool_calls = [
        ChatCompletionMessageToolCall(
            id=f"fake_{next(_ids)}", type="function",
            function=Function(name=name, arguments=args if isinstance(args, str) else json.dumps(args)))
        for name, args in calls
    ]
    return _response(Message(role="assistant", content=thought or None, tool_calls=tool_calls),
                     "tool_calls")


# --- the fake model -----------------------------------------------------------

def _get(obj, key):
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _called_tools(message) -> list[tuple[str, str]]:
    return [(_get(_get(c, "function"), "name"), _get(_get(c, "function"), "arguments"))
            for c in _get(message, "tool_calls") or []]


def respond(history: list, tools: list | None, tool_choice: dict | None) -> ModelResponse:
    # FAKE_DELAY=1 makes demo mode feel like a real model (the UI timeline shows its spinners)
    time.sleep(float(os.getenv("FAKE_DELAY") or 0))
    # The request this run is answering: the last user message the loop did not write itself
    start = max(i for i, m in enumerate(history)
                if _get(m, "role") == "user" and not (_get(m, "content") or "").startswith(AUTO_PREFIX))
    request = _get(history[start], "content") or ""
    called = [name for m in history[start + 1:] if _get(m, "role") == "assistant"
              for name, _ in _called_tools(m)]

    if tool_choice:  # the loop is forcing a specific tool
        return _call(tool_choice["function"]["name"])
    if "=== JOB POST" in request:  # first run: follow the script
        for name in ORDER:
            if name not in called:
                return _call(name)
        return text_reply("The application pack is saved.")
    return _follow_up(history, request, called)


def _call(name: str, args: dict | None = None) -> ModelResponse:
    return tool_reply(THOUGHTS[name], [(name, args or ARGS[name])])


def _follow_up(history: list, request: str, called: list) -> ModelResponse:
    if request.startswith(FACTS_PREFIX):
        return _use_facts(history, request, called)
    shorter = re.search(r"answer\s*(\d+)", request.lower())
    if shorter and "shorter" in request.lower() and "save_application_pack" not in called:
        pack = json.loads(json.dumps(_last_saved_pack(history)))  # deep copy
        n = int(shorter.group(1))
        if 1 <= n <= len(pack["answers"]):
            pack["answers"][n - 1]["answer"] = _shorten(pack["answers"][n - 1]["answer"])
            return tool_reply(f"I'll shorten answer {n} and save the full updated pack.",
                              [("save_application_pack", pack)])
    return text_reply("Your biggest gap is LLM evaluation: your resume has no test set or scoring "
                      "for model output. Building a small one for the delivery-email parser is the "
                      "quickest way to close it.")


def _use_facts(history: list, request: str, called: list) -> ModelResponse:
    """Rewrite the answers and the letter: each filled [ADD: hint] becomes the candidate's value."""
    if "save_application_pack" in called:
        return text_reply("Done: your details are in the answers and the cover letter.")
    facts = {hint.strip(): value.strip() for hint, value in re.findall(r'^- For "(.+?)": (.+)$', request, re.M)}
    fill = lambda text: re.sub(r"\[ADD:\s*(.*?)\]", lambda m: facts.get(m.group(1).strip(), m.group(0)), text)  # noqa: E731
    pack = json.loads(json.dumps(_last_saved_pack(history)))  # deep copy
    for answer in pack["answers"]:
        answer["answer"] = fill(answer["answer"])
    pack["cover_letter"] = fill(pack["cover_letter"])
    return tool_reply("I'll rewrite your answers and the cover letter with your details and save the full pack.",
                      [("save_application_pack", pack)])


def _last_saved_pack(history: list) -> dict:
    for message in reversed(history):
        if _get(message, "role") == "assistant":
            for name, arguments in _called_tools(message):
                if name == "save_application_pack":
                    return json.loads(arguments)
    return PACK


def _shorten(text: str) -> str:
    """Keep whole sentences until about 40 words."""
    kept, words = [], 0
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if kept and words + len(sentence.split()) > 40:
            break
        kept.append(sentence)
        words += len(sentence.split())
    return " ".join(kept)
