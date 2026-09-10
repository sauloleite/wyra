"""Markdown into a fine-tuning dataset, with no model and no API key.

python examples/build_markdown.py
"""

from __future__ import annotations

from pathlib import Path

from wyra import build_dataset

HERE = Path(__file__).parent

result = build_dataset(
    HERE / "data" / "clean_code.md",
    HERE / "out" / "markdown",
    lang="pt-br",
    system_prompt="Você é um tutor de boas práticas de programação. Responda em português.",
    validation_fraction=0.1,
)

print(result.report.summary())
print(f"train:    {result.train_path}")
print(f"manifest: {result.manifest_path}")
print()
print("first example:")
first = result.train[0]
for message in first.messages:
    print(f"  {message.role.value}: {message.content[:90]}...")
