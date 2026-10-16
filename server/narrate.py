"""narrate_path: the LLM's short story of a progression.

Calls Ollama's /api/chat with PathStory's JSON schema as the reply format. The model
comes from LLM_MODEL (default qwen3:4b-instruct; an "ollama/" prefix is accepted),
the server from OLLAMA_URL.
"""

from __future__ import annotations

import json
import os

import httpx

from server.models import PathStory

OLLAMA_URL: str = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
LLM_MODEL: str = os.environ.get("LLM_MODEL", "qwen3:4b-instruct").removeprefix("ollama/")

# what the model is told: the task, then what each argument means
INSTRUCTION = (
    "You are a warm, precise composition teacher. A composer is auditioning a chord "
    "progression; characterise it for them. Only use the facts given."
)
ARGS = {
    "key": "The key, e.g. 'C major'.",
    "progression": "The progression as Roman numerals with chord symbols.",
    "cadence": "The cadence the path ends with, or 'open' if none.",
    "moves": "The harmonic-syntax rule behind each move.",
    "voice_leading": "Voice-leading observations the analysis engine made.",
}


def narrate_path(
    key: str, progression: str, cadence: str, moves: list[str], voice_leading: list[str]
) -> PathStory:
    values = {"key": key, "progression": progression, "cadence": cadence, "moves": moves, "voice_leading": voice_leading}
    facts = "\n".join(f"{name} ({ARGS[name]}): {json.dumps(v, ensure_ascii=False)}" for name, v in values.items())
    fields = "\n".join(f"{name}: {f.description}" for name, f in PathStory.model_fields.items())
    resp = httpx.post(
        OLLAMA_URL + "/api/chat",
        json={
            "model": LLM_MODEL,
            "stream": False,
            "format": PathStory.model_json_schema(),
            "options": {"temperature": 0.6},
            "messages": [
                {"role": "system", "content": INSTRUCTION + "\n\nReply with JSON with these fields:\n" + fields},
                {"role": "user", "content": facts},
            ],
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    return PathStory.model_validate_json(resp.json()["message"]["content"])
