"""Check that the configured model and key work: python scripts/smoke_llm.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import llm  # noqa: E402

try:
    response = llm.chat("You are a terse assistant.",
                        [{"role": "user", "content": "Reply with exactly: key works"}],
                        on_wait=print)
except llm.LLMConfigError as err:
    sys.exit(f"Config problem: {err}")
except Exception as err:
    # Provider error text does not include your key
    sys.exit(f"Call failed ({type(err).__name__}): {err}")

print("Answer:", response.choices[0].message.content)
print("Model:", response.model, "(configured:", llm.get_model() + ")")
