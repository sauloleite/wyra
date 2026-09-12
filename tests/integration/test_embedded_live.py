"""Contract test for the embedded engine, against real weights. Skipped unless enabled.

Run with::

    pip install 'wyra[local]'
    wyra setup --download qwen2.5-0.5b
    WYRA_LIVE_TESTS=1 WYRA_LIVE_MODEL=qwen2.5-0.5b pytest -m live

It needs the extra installed and the weights already in the cache, so it never downloads
gigabytes on its own. Everything else in the suite runs offline.
"""

from __future__ import annotations

import importlib.util
import json
import os

import pytest

from wyra.domain import Document
from wyra.generators import QAPairGenerator
from wyra.prompts import PT_BR
from wyra.providers import modelstore
from wyra.providers.embedded import EmbeddedProvider
from wyra.validation import validate_records

MODEL = os.environ.get("WYRA_LIVE_MODEL", modelstore.DEFAULT_MODEL)
TEXT = (
    "A regra do escoteiro diz para deixar o código mais limpo do que estava antes de "
    "você mexer nele. Refatorar precisa virar um hábito diário, feito enquanto a lógica "
    "ainda está fresca na cabeça, e não uma reescrita rara e arriscada."
)


def _ready() -> bool:
    if not os.environ.get("WYRA_LIVE_TESTS"):
        return False
    if importlib.util.find_spec("onnxruntime_genai") is None:
        return False
    return MODEL in modelstore.CATALOG and modelstore.is_cached(MODEL)


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not _ready(),
        reason="set WYRA_LIVE_TESTS=1, install wyra[local] and download a model first",
    ),
]


def test_constrained_decoding_returns_schema_shaped_json() -> None:
    provider = EmbeddedProvider(MODEL, max_new_tokens=512)
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    from wyra.domain import Message, Role

    completion = provider.complete(
        [Message(Role.USER, "O que é refatorar? Responda em JSON.")],
        json_schema=schema,
        temperature=0.0,
    )
    payload = json.loads(completion.text)
    assert isinstance(payload.get("answer"), str) and payload["answer"].strip()
    assert completion.output_tokens and completion.output_tokens > 0


def test_pairs_generated_locally_are_valid_records() -> None:
    generator = QAPairGenerator(
        EmbeddedProvider(MODEL, max_new_tokens=768), prompts=PT_BR, per_chunk=2, min_grounding=0.0
    )
    examples = list(generator.generate(Document(TEXT, source="live.txt")))
    assert examples, "the local model returned no usable pairs"

    records = [
        (i, {"messages": [{"role": m.role.value, "content": m.content} for m in e.messages]})
        for i, e in enumerate(examples, start=1)
    ]
    report = validate_records(records, "openai-chat")
    assert report.ok, report.summary()
