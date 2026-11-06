"""System prompt, Session, and the hand-written agent loop."""
from dataclasses import dataclass, field


@dataclass
class Session:
    """Everything one application run knows. Tools read and write it; no globals."""
    job_text: str = ""
    resume_text: str = ""
    questions: list[str] = field(default_factory=list)
    fit: dict | None = None   # last calculate_fit_score result
    pack: dict | None = None  # saved application pack
