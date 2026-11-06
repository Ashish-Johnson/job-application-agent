"""Run the agent on samples/ and print its events live: python scripts/run_cli.py
Works with FAKE_LLM=1 (no model) or with the model set in .env."""
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import agent  # noqa: E402


def show(event: dict) -> None:
    stamp = f"[{event['t']:6.1f}s]"
    kind = event["type"]
    if kind == "thought":
        print(f"{stamp} {event['text']}")
    elif kind == "tool_start":
        print(f"{stamp}   -> {event['label']}")
    elif kind == "tool_end":
        print(f"{stamp}   {'ok' if event['ok'] else 'FAILED'}: {event['summary']}")
        for issue in event.get("issues", []):
            print(f"{stamp}       - {issue}")
    elif kind == "wait":
        print(f"{stamp} WAIT: {event['message']}")
    elif kind == "reply":
        print(f"{stamp} REPLY: {event['text']}")
    elif kind == "error":
        print(f"{stamp} ERROR: {event['message']}")
    elif kind == "result":
        print(f"{stamp} RESULT: pack saved")


def show_pack(pack: dict) -> None:
    print(f"\n=== {pack['role']} at {pack['company']}: {pack['score']}/100, {pack['verdict']} ===")
    print(pack["summary"])
    print("\nStrengths:")
    for s in pack["strengths"]:
        print(f"  - {s['strength']}\n      evidence: {s['evidence']}")
    print("\nGaps:")
    for g in pack["gaps"]:
        print(f"  - {g['gap']}\n      fix: {g['how_to_address']}")
    for i, a in enumerate(pack["answers"], 1):
        print(f"\nQ{i}. {a['question']}\n{a['answer']}")
    print(f"\nCover note:\n{pack['cover_note']}")
    for warning in pack.get("warnings", []):
        print(f"\nWARNING (left after honesty-check retries): {warning}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to a narrow codec
    samples = ROOT / "samples"
    session = agent.Session(
        job_text=(samples / "job_post.txt").read_text(encoding="utf-8"),
        resume_text=(samples / "resume.txt").read_text(encoding="utf-8"),
        questions=[q for q in (samples / "questions.txt").read_text(encoding="utf-8").splitlines() if q.strip()],
    )
    started = time.monotonic()
    # Same pattern the web server will use: the agent runs in a thread, we read its events
    worker = threading.Thread(target=agent.run_turn, args=(session, agent.build_task_message(session)))
    worker.start()
    seen = 0
    while True:
        running = worker.is_alive()  # read before fetching, so the last events are not missed
        for event in session.events_since(seen):
            show(event)
            seen += 1
        if not running:
            break
        time.sleep(0.2)

    if session.pack:
        show_pack(session.pack)
    print(f"\nStatus: {session.status} · Model calls: {session.model_calls} "
          f"· Total time: {time.monotonic() - started:.1f}s")
    return 0 if session.status == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
