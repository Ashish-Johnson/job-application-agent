"""Start the app: uvicorn on 127.0.0.1:8000 (local only).  python run.py [--no-browser]"""
import sys
import threading
import webbrowser

import uvicorn

from app import llm

HOST, PORT = "127.0.0.1", 8000  # never 0.0.0.0: the app must stay on this machine


def main() -> None:
    url = f"http://{HOST}:{PORT}"
    model = "fake (FAKE_LLM=1, no real model)" if llm.fake_enabled() else llm.get_model()
    print(f"Job Application Agent: {url}")
    print(f"Model: {model}")
    if llm.fake_enabled() or model.startswith("ollama/"):
        print("Privacy: runs on your machine and nothing is sent anywhere.")
    else:
        print(f"Privacy: runs on your machine; only the text you submit is sent to {model.split('/')[0]}.")
    try:
        llm.check_config()
    except llm.LLMConfigError as err:
        print(f"WARNING: {err}")
    if "--no-browser" not in sys.argv:
        threading.Timer(1.0, webbrowser.open, args=[url]).start()
    uvicorn.run("app.main:app", host=HOST, port=PORT)


if __name__ == "__main__":
    main()
