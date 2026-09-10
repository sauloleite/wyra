from __future__ import annotations

import json
from pathlib import Path

import pytest

from wyra.domain import Example
from wyra.errors import FormatError
from wyra.writers import InMemoryWriter, JsonlWriter


def test_jsonl_writer_writes_one_utf8_line_per_example(
    tmp_path: Path, sample_examples: list[Example]
) -> None:
    writer = JsonlWriter()
    assert writer.format == "openai-chat"
    destination = tmp_path / "nested" / "train.jsonl"
    assert writer.write(sample_examples, destination) == 3
    raw = destination.read_bytes().decode("utf-8")
    assert raw.endswith("\n") and raw.count("\n") == 3
    assert "Não repita" in raw and "\\u00e3" not in raw
    records = [json.loads(line) for line in raw.splitlines()]
    assert records[0]["messages"][0] == {"role": "system", "content": "Você é um tutor."}


def test_jsonl_writer_other_formats(tmp_path: Path, sample_examples: list[Example]) -> None:
    path = tmp_path / "alpaca.jsonl"
    JsonlWriter("alpaca").write(sample_examples, path)
    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert first["instruction"] == "O que é DRY?" and first["system"] == "Você é um tutor."
    with pytest.raises(FormatError):
        JsonlWriter("nope")


def test_in_memory_writer_keeps_records(sample_examples: list[Example]) -> None:
    writer = InMemoryWriter("sharegpt")
    assert writer.write(sample_examples[:2], "train.jsonl") == 2
    assert writer.write([], "valid.jsonl") == 0
    assert list(writer.records) == ["train.jsonl", "valid.jsonl"]
    assert writer.records["train.jsonl"][1]["conversations"][0]["from"] == "human"
