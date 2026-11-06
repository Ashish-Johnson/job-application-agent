"""Five tools + their schemas (OpenAI function-calling format).

Every tool is a plain function: tool(session, **args) -> dict.
On failure it returns {"error": "..."} instead of raising, so the model can read
the error and adapt.
"""
import ipaddress
import os
import re
import socket
from urllib.parse import quote, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

# Wikimedia rejects (403) user agents without a contact URL or email.
# Set CONTACT_URL in .env to your own repo URL; this default is only a placeholder.
DEFAULT_CONTACT = "https://github.com/example/job-application-agent"


def user_agent() -> str:
    contact = os.getenv("CONTACT_URL") or DEFAULT_CONTACT
    return f"JobApplicationAgent/0.1 ({contact}; local educational project) python-httpx"


MAX_PAGE_CHARS = 8000
MIN_PAGE_CHARS = 200
MAX_REDIRECTS = 3

MAX_HONESTY_REJECTIONS = 2  # rejected saves per turn; after that we save with warnings

# Feelings and motivations only the candidate can supply. Allowed only inside an
# [ADD: ...] placeholder, unless the resume itself contains the phrase.
FEELING_PHRASES = ["I enjoy", "I love", "excited", "passionate", "thrive",
                   "my favourite", "my favorite", "dream"]

IMPORTANCE_WEIGHT = {"must": 2, "nice": 1}
MATCH_CREDIT = {"strong": 1, "partial": 0.5, "missing": 0}


# --- 1. fetch_webpage --------------------------------------------------------

def check_public_url(url: str) -> str | None:
    """Return an error message if the URL is not a public http(s) address, else None."""
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return "Only http(s) URLs are allowed."
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(parts.hostname, None)}
    except socket.gaierror:
        return f"Could not resolve host '{parts.hostname}'."
    for address in addresses:
        ip = ipaddress.ip_address(address.split("%")[0])  # strip IPv6 zone id
        if not ip.is_global:  # private, loopback, link-local, reserved...
            return "Refusing to fetch localhost or private network addresses."
    return None


def fetch_webpage(session, url: str) -> dict:
    headers = {"User-Agent": user_agent()}
    try:
        # Redirects are followed by hand so every hop gets the safety check.
        with httpx.Client(timeout=15, follow_redirects=False, headers=headers) as client:
            for _ in range(MAX_REDIRECTS + 1):
                problem = check_public_url(url)
                if problem:
                    return {"error": problem}
                response = client.get(url)
                if not response.is_redirect:
                    break
                url = urljoin(url, response.headers.get("location", ""))
            else:
                return {"error": "Too many redirects."}
    except httpx.HTTPError as err:
        return {"error": f"Could not fetch the page: {err}"}

    if response.status_code >= 400:
        return {"error": f"The page returned HTTP {response.status_code}."}
    if "text" not in response.headers.get("content-type", ""):  # covers text/html
        return {"error": "The URL is not a web page (unsupported content type)."}

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    text = re.sub(r"\n{2,}", "\n", soup.get_text("\n", strip=True))
    if len(text) < MIN_PAGE_CHARS:
        return {"error": "Almost no text was found; the page probably needs JavaScript. "
                         "Ask the user to paste the job post text instead."}
    return {"url": url, "text": text[:MAX_PAGE_CHARS], "truncated": len(text) > MAX_PAGE_CHARS}


# --- 2. lookup_company -------------------------------------------------------

def lookup_company(session, name: str) -> dict:
    headers = {"User-Agent": user_agent()}
    try:
        with httpx.Client(timeout=15, headers=headers) as client:
            search = client.get("https://en.wikipedia.org/w/api.php", params={
                "action": "query", "list": "search", "srsearch": name,
                "srlimit": 1, "format": "json"})
            search.raise_for_status()
            hits = search.json()["query"]["search"]
            if not hits:
                return {"found": False}
            title = hits[0]["title"]
            page = client.get("https://en.wikipedia.org/api/rest_v1/page/summary/"
                              + quote(title.replace(" ", "_"), safe=""))
            page.raise_for_status()
            data = page.json()
    except (httpx.HTTPError, KeyError, ValueError) as err:
        return {"error": f"Wikipedia lookup failed: {err}"}
    if data.get("type") == "disambiguation":
        return {"found": False, "note": f"'{title}' is a disambiguation page, not a company."}
    return {
        "found": True,
        "title": data.get("title", title),
        "description": data.get("description", ""),
        "summary": data.get("extract", "")[:1500],
        "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
    }


# --- 3. check_resume_for -----------------------------------------------------

def _skill_pattern(alternative: str) -> re.Pattern | None:
    """Regex that matches a skill ignoring case and . - _ spaces ("Node.js" ~ "nodejs").

    Letters/digits (plus + and #, so C, C++ and C# stay distinct) must match in
    order; the boundaries stop "Java" matching inside "JavaScript".
    """
    chars = [c for c in alternative.lower() if c.isalnum() or c in "+#"]
    if not chars:
        return None
    body = r"[\s._-]*".join(re.escape(c) for c in chars)
    return re.compile(rf"(?<![a-z0-9+#]){body}(?![a-z0-9+#])", re.IGNORECASE)


def check_resume_for(session, skills: list[str]) -> dict:
    if not isinstance(skills, list) or not skills:
        return {"error": "skills must be a non-empty list of strings."}
    lines = [line.strip() for line in session.resume_text.splitlines() if line.strip()]
    results = []
    for skill in skills:
        patterns = [p for p in map(_skill_pattern, str(skill).split("/")) if p]
        evidence = [line[:200] for line in lines if any(p.search(line) for p in patterns)][:2]
        results.append({"skill": skill, "found": bool(evidence), "evidence": evidence})
    return {"results": results}


# --- 4. calculate_fit_score --------------------------------------------------

def calculate_fit_score(session, requirements: list[dict]) -> dict:
    if not isinstance(requirements, list) or not requirements:
        return {"error": "requirements must be a non-empty list."}
    breakdown, earned, possible = [], 0.0, 0
    for item in requirements:
        importance, match = item.get("importance"), item.get("match")
        if importance not in IMPORTANCE_WEIGHT or match not in MATCH_CREDIT:
            return {"error": f"Bad entry {item!r}: importance must be 'must' or 'nice'; "
                             "match must be 'strong', 'partial' or 'missing'."}
        weight = IMPORTANCE_WEIGHT[importance]
        points = weight * MATCH_CREDIT[match]
        earned += points
        possible += weight
        breakdown.append({"requirement": item.get("requirement", ""), "importance": importance,
                          "match": match, "points": points, "possible": weight})

    score = round(100 * earned / possible)
    missing_must = [b["requirement"] for b in breakdown
                    if b["importance"] == "must" and b["match"] == "missing"]
    if score >= 75 and not missing_must:
        verdict = "Strong fit"
    elif score >= 50:
        verdict = "Good fit with gaps"
    else:
        verdict = "Stretch role"
    session.fit = {"score": score, "verdict": verdict,
                   "missing_must_haves": missing_must, "breakdown": breakdown}
    return session.fit


# --- 5. save_application_pack ------------------------------------------------

def _strip_question_numbers(answers: list) -> list:
    """Models copy the "1. " numbering from the task message into the question text."""
    return [dict(a, question=re.sub(r"^\s*\d+[.)]\s*", "", a["question"]))
            if isinstance(a, dict) and isinstance(a.get("question"), str) else a
            for a in answers]


def _norm(text: str) -> str:
    """Lowercase, straighten curly quotes, collapse whitespace."""
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return " ".join(text.lower().split())


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in re.findall(r"\d+(?:[.,]\d+)*", text)}


def _around(text: str, match: re.Match) -> str:
    return " ".join(text[max(0, match.start() - 25):match.end() + 25].split())


def honesty_issues(session, fields: dict) -> list[str]:
    """Deterministic checks on a pack before it is saved. Returns specific, fixable issues."""
    issues = []
    resume = _norm(session.resume_text)
    allowed_numbers = _numbers(session.resume_text) | _numbers(session.job_text)

    # 1. Every strength's evidence must be a real quote from the resume
    for i, item in enumerate(fields["strengths"], 1):
        evidence = item.get("evidence", "") if isinstance(item, dict) else ""
        # "..." lets the model skip words inside one quote; each piece must still match
        pieces = [_norm(p).strip(" \"'-•") for p in re.split(r"\.\.\.|…", evidence)]
        pieces = [p for p in pieces if p]
        if not pieces or not all(p in resume for p in pieces):
            issues.append(f'Strength {i}: the evidence is not a quote from the resume: "{evidence[:80]}". '
                          "Copy the resume text exactly.")

    # 2 and 3. Numbers and feelings in everything the candidate would say or send.
    # (label, text, check_numbers): how_to_address is advice ("practise for 4 weeks"), not a
    # claim about the candidate, so only its feelings are checked.
    texts = [("summary", fields["summary"], True), ("cover note", fields["cover_note"], True)]
    for i, item in enumerate(fields["strengths"], 1):
        texts.append((f"strength {i}", item.get("strength", "") if isinstance(item, dict) else "", True))
    for i, item in enumerate(fields["gaps"], 1):
        for key in ("gap", "how_to_address"):
            texts.append((f"gap {i} ({key})", item.get(key, "") if isinstance(item, dict) else "",
                          key != "how_to_address"))
    for i, item in enumerate(fields["answers"], 1):
        texts.append((f"answer {i}", item.get("answer", "") if isinstance(item, dict) else "", True))

    for label, text, check_numbers in texts:
        text = re.sub(r"\[ADD:[^\]]*\]", " ", str(text))  # placeholders may hold anything
        reported = set()
        for match in re.finditer(r"\d+(?:[.,]\d+)*", text if check_numbers else ""):
            number = match.group().replace(",", "")
            if number not in allowed_numbers and number not in reported:
                reported.add(number)
                issues.append(f'{label}: the number "{number}" is not in the resume or job post '
                              f'(in "...{_around(text, match)}..."). Remove it or use an [ADD: ...] placeholder.')
        for phrase in FEELING_PHRASES:
            match = re.search(r"\b" + re.escape(phrase), text, re.IGNORECASE)
            if match and _norm(phrase) not in resume:
                issues.append(f'{label}: "{phrase}" states a feeling or motivation the resume does not '
                              f'support (in "...{_around(text, match)}..."). Only the candidate can supply '
                              "that: replace it with an [ADD: ...] placeholder.")
    return issues[:15]


def save_application_pack(session, role: str, company: str, summary: str, strengths: list,
                          gaps: list, answers: list, cover_note: str) -> dict:
    if session.fit is None:
        return {"error": "No fit score yet. Call calculate_fit_score first, then save the pack."}
    answers = _strip_question_numbers(answers)
    fields = {"summary": summary, "cover_note": cover_note,
              "strengths": strengths, "gaps": gaps, "answers": answers}

    issues = honesty_issues(session, fields)
    if issues and session.honesty_rejections < MAX_HONESTY_REJECTIONS:
        session.honesty_rejections += 1
        return {"error": f"Honesty check failed with {len(issues)} issue(s); nothing was saved. "
                         "Fix exactly these and call save_application_pack again with the FULL pack.",
                "issues": issues}

    session.pack = {
        "role": role, "company": company, "summary": summary,
        "strengths": strengths, "gaps": gaps, "answers": answers, "cover_note": cover_note,
        # Always taken from the calculator, never from the model, so the score can't be invented.
        "score": session.fit["score"],
        "verdict": session.fit["verdict"],
        "missing_must_haves": session.fit["missing_must_haves"],
        "breakdown": session.fit["breakdown"],
        # Issues left after the retries ran out; the UI shows them to the user
        "warnings": issues,
    }
    return {"saved": True, "score": session.fit["score"], "verdict": session.fit["verdict"],
            "warnings": len(issues)}


# --- Dispatch and UI text ----------------------------------------------------

FUNCTIONS = {
    "fetch_webpage": fetch_webpage,
    "lookup_company": lookup_company,
    "check_resume_for": check_resume_for,
    "calculate_fit_score": calculate_fit_score,
    "save_application_pack": save_application_pack,
}


def run_tool(session, name: str, args: dict) -> dict:
    """Run one tool call from the model. Never raises."""
    function = FUNCTIONS.get(name)
    if function is None:
        return {"error": f"Unknown tool '{name}'. Available: {', '.join(FUNCTIONS)}."}
    try:
        return function(session, **args)
    except TypeError as err:  # wrong or missing arguments
        return {"error": f"Bad arguments for {name}: {err}"}
    except Exception as err:  # a tool bug must not crash the whole run
        return {"error": f"{name} failed: {err}"}


# Human-friendly text for the UI timeline: label(args) before, summary(result) after.
LABELS = {
    "fetch_webpage": lambda a: f"Reading {a.get('url', 'the job post')}",
    "lookup_company": lambda a: f"Looking up {a.get('name', 'the company')}",
    "check_resume_for": lambda a: f"Checking your resume for {len(a.get('skills') or [])} skills",
    "calculate_fit_score": lambda a: f"Scoring fit against {len(a.get('requirements') or [])} requirements",
    "save_application_pack": lambda a: "Writing your application pack",
}


def _summary_check_resume(result):
    found = [r for r in result["results"] if r["found"]]
    missing = [r["skill"] for r in result["results"] if not r["found"]]
    text = f"Found {len(found)} of {len(result['results'])}"
    return text + (f" · missing: {', '.join(missing)}" if missing else "")


SUMMARIES = {
    "fetch_webpage": lambda r: f"Read {len(r['text'])} characters",
    "lookup_company": lambda r: f"Found: {r['title']}" if r["found"] else "No Wikipedia page found",
    "check_resume_for": _summary_check_resume,
    "calculate_fit_score": lambda r: f"{r['score']}/100 · {r['verdict']}",
    "save_application_pack": lambda r: "Application pack saved" + (
        f" with {r['warnings']} warning(s)" if r.get("warnings") else ""),
}


def label_for(name: str, args: dict) -> str:
    try:
        return LABELS[name](args)
    except Exception:
        return f"Running {name}"


def summary_for(name: str, result: dict) -> str:
    if "issues" in result:
        n = len(result["issues"])
        return f"Honesty check: {n} issue{'s' if n != 1 else ''}, asking the agent to fix {'them' if n != 1 else 'it'}"
    if "error" in result:
        return f"Failed: {result['error']}"
    try:
        return SUMMARIES[name](result)
    except Exception:
        return "Done"


# --- Schemas the model sees --------------------------------------------------

def _tool(name: str, description: str, properties: dict, required: list) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required}}}


SCHEMAS = [
    _tool(
        "fetch_webpage",
        "Download a public web page and return its readable text (up to 8,000 characters). "
        "Use it ONLY when the user gave a URL for the job post instead of pasting the text. "
        "Do not use it for localhost or private addresses. If it reports that the page needs "
        "JavaScript, ask the user to paste the job post text; do not guess its contents.",
        {"url": {"type": "string", "description": "Full http(s) URL of the job post."}},
        ["url"]),
    _tool(
        "lookup_company",
        "Look up a company on Wikipedia and return its title, short description and summary. "
        "Use it once to understand what the employer does, for the 'why this company' answer "
        "and the cover note. Small companies often have no page: if found is false, continue "
        "using only what the job post says. Check the description matches the company in the "
        "job post, since search can return a different organisation with a similar name. "
        "Never state facts about the company that this tool or the job post did not provide.",
        {"name": {"type": "string", "description": "Company name, e.g. 'Stripe'."}},
        ["name"]),
    _tool(
        "check_resume_for",
        "Search the candidate's resume for specific skills and return the exact resume lines "
        "that mention each one. It is deterministic text search, not judgement: found=false "
        "means the resume does not mention it. Call it with the job's key skills BEFORE rating "
        "any match, and quote only the evidence it returns. Separate alternatives with '/', "
        "e.g. 'Node.js/Express'. Matching ignores case and punctuation.",
        {"skills": {"type": "array", "items": {"type": "string"},
                    "description": "Skills or technologies, e.g. ['Python', 'FastAPI', 'Docker']."}},
        ["skills"]),
    _tool(
        "calculate_fit_score",
        "Compute the 0-100 fit score and verdict from your requirement ratings. Call it exactly "
        "once, after checking the resume, listing EVERY must-have and nice-to-have from the job "
        "post. Rate match as 'strong' only with resume evidence, 'partial' for related or weaker "
        "evidence, and 'missing' when the resume has none. Do not compute or guess the score "
        "yourself; the saved pack always uses this tool's result.",
        {"requirements": {"type": "array", "items": {"type": "object", "properties": {
            "requirement": {"type": "string", "description": "Short requirement text."},
            "importance": {"type": "string", "enum": ["must", "nice"]},
            "match": {"type": "string", "enum": ["strong", "partial", "missing"]}},
            "required": ["requirement", "importance", "match"]}}},
        ["requirements"]),
    _tool(
        "save_application_pack",
        "Save the finished application pack. Call it last, only after calculate_fit_score. "
        "Every claim about the candidate must come from the resume; where a fact is missing "
        "write a placeholder like [ADD: metric for this achievement] instead of inventing it. "
        "Include a drafted answer for every application question and a short cover note. "
        "Do not include a score: it is attached automatically. An automatic honesty check runs "
        "first and rejects the save, returning a list of issues, if a strength's evidence is not an "
        "exact quote from the resume, if any number is not in the resume or job post (including in "
        "gap fixes: describe without numbers or use [ADD: ...]), or if a feeling or motivation "
        "('I enjoy', 'excited', 'passionate'...) is written outside an [ADD: ...] placeholder. "
        "If issues come back, fix exactly those and call this tool again with the full pack.",
        {"role": {"type": "string"},
         "company": {"type": "string"},
         "summary": {"type": "string", "description": "2-3 sentence overview of the fit."},
         "strengths": {"type": "array", "items": {"type": "object", "properties": {
             "strength": {"type": "string"},
             "evidence": {"type": "string", "description": "Resume text that backs it."}},
             "required": ["strength", "evidence"]}},
         "gaps": {"type": "array", "items": {"type": "object", "properties": {
             "gap": {"type": "string"},
             "how_to_address": {"type": "string"}},
             "required": ["gap", "how_to_address"]}},
         "answers": {"type": "array", "items": {"type": "object", "properties": {
             "question": {"type": "string"},
             "answer": {"type": "string"}},
             "required": ["question", "answer"]}},
         "cover_note": {"type": "string", "description": "Short cover note, under 150 words."}},
        ["role", "company", "summary", "strengths", "gaps", "answers", "cover_note"]),
]
