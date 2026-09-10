"""Contract test against a real Ollama server. Skipped unless explicitly enabled.

Run with::

    WYRA_LIVE_TESTS=1 pytest -m live

It needs the server running and enough free memory for the model (about 2.1 GiB for
llama3.2:1b). Everything else in the suite runs without a network.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pytest

from wyra.domain import Document
from wyra.generators import QAPairGenerator
from wyra.prompts import PT_BR
from wyra.providers.ollama import OllamaProvider
from wyra.validation import validate_records

pytestmark = pytest.mark.live

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
MODEL = os.environ.get("WYRA_LIVE_MODEL", "llama3.2:1b")
TEXT = (
    "A regra do escoteiro diz para deixar o código mais limpo do que estava antes de "
    "você mexer nele. Refatorar precisa virar um hábito diário, feito enquanto a lógica "
    "ainda está fresca na cabeça, e não uma reescrita rara e arriscada."
)


def _server_is_up() -> bool:
    if not os.environ.get("WYRA_LIVE_TESTS"):
        return False
    url = HOST if "://" in HOST else f"http://{HOST}"
    try:
        with urllib.request.urlopen(f"{url}/api/version", timeout=3):
            return True
    except (urllib.error.URLError, OSError):
        return False


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not _server_is_up(), reason="set WYRA_LIVE_TESTS=1 and run Ollama"),
]


def test_structured_output_produces_valid_records() -> None:
    provider = OllamaProvider(MODEL, timeout=300)
    generator = QAPairGenerator(provider, prompts=PT_BR, per_chunk=2, min_grounding=0.0)
    examples = list(generator.generate(Document(TEXT, source="live.txt")))

    assert examples, "the model returned no usable pairs"
    records = [
        (i, {"messages": [{"role": m.role.value, "content": m.content} for m in e.messages]})
        for i, e in enumerate(examples, start=1)
    ]
    report = validate_records(records, "openai-chat")
    assert report.ok, report.summary()


def test_the_format_field_constrains_decoding() -> None:
    provider = OllamaProvider(MODEL, timeout=300)
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    completion = provider.complete(
        [__import__("wyra").Message("user", "Responda em JSON: o que é refatorar?")],
        json_schema=schema,
    )
    payload = json.loads(completion.text)
    assert isinstance(payload.get("answer"), str)
    assert completion.output_tokens is None or completion.output_tokens > 0
