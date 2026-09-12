"""Free prose into question/answer pairs with a local model. No API key, no cost.

Needs Ollama running and a model pulled::

    ollama pull llama3.2:3b
    python examples/build_ollama.py

A small model writes weak pairs. Read a sample before training on them: the schema
validator guarantees the shape, never the meaning.
"""

from __future__ import annotations

import os
from pathlib import Path

from wyra import build_dataset
from wyra.chunking import ParagraphChunker
from wyra.generators import QAPairGenerator
from wyra.prompts import PT_BR
from wyra.providers import create

HERE = Path(__file__).parent

generator = QAPairGenerator(
    create("ollama", model=os.environ.get("WYRA_MODEL", "llama3.2:3b")),
    prompts=PT_BR,
    per_chunk=3,
    min_grounding=0.2,
)

result = build_dataset(
    HERE / "data" / "clean_code.md",
    HERE / "out" / "ollama",
    generator=generator,
    chunker=ParagraphChunker(max_chars=800),
    validation_fraction=0.1,
    on_error="skip",
)

print(result.report.summary())
print(f"train: {result.train_path}")
for example in result.train[:3]:
    print(f"\nQ: {example.messages[0].content}")
    print(f"A: {example.messages[1].content[:160]}")
