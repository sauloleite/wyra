from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from wyra.domain import Document, Example
from wyra.errors import ValidationError
from wyra.readers import read_document, read_documents, read_examples, read_jsonl_lines


def test_read_document_records_sha256_and_strips_bom(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_bytes("﻿Olá".encode())
    doc = read_document(path)
    assert doc.text == "Olá"
    assert doc.source == str(path)
    assert doc.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert doc.index == 0


def test_read_documents_expands_globs_dirs_and_keeps_order(tmp_path: Path) -> None:
    (tmp_path / "b.md").write_text("b", encoding="utf-8")
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.txt").write_text("c", encoding="utf-8")

    docs = read_documents(str(tmp_path / "*.md"))
    assert [d.text for d in docs] == ["a", "b"]

    docs = read_documents(tmp_path)
    assert [d.text for d in docs] == ["a", "b", "c"]

    inline = Document("inline")
    docs = read_documents([inline, tmp_path / "a.md", tmp_path / "a.md"])
    assert docs == [inline, docs[1]]
    assert docs[1].text == "a"

    with pytest.raises(FileNotFoundError):
        read_documents(tmp_path / "missing.md")
    with pytest.raises(FileNotFoundError):
        read_documents(str(tmp_path / "*.nope"))


def test_read_jsonl_lines_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    path.write_text('{"a": 1}\n\n  \n{"b": 2}\n', encoding="utf-8")
    assert list(read_jsonl_lines(path)) == [(1, '{"a": 1}'), (4, '{"b": 2}')]


def test_read_examples_auto_detects_and_reports_line(fixtures: Path, tmp_path: Path) -> None:
    examples = list(read_examples(fixtures / "sharegpt.jsonl"))
    assert len(examples) == 2 and examples[0].system == "Seja breve."
    alpaca = list(read_examples(fixtures / "alpaca.jsonl", input_format="alpaca"))
    assert alpaca[0] == Example.qa("Traduza para o inglês.\n\ncódigo limpo", "clean code")

    with pytest.raises(ValidationError) as info:
        list(read_examples(fixtures / "bad.jsonl"))
    assert info.value.index == 1 and "data_type" in info.value.issues

    broken = tmp_path / "broken.jsonl"
    broken.write_text(
        '{"messages": [{"role": "user", "content": "q"}, '
        '{"role": "assistant", "content": "a"}]}\n{oops\n',
        encoding="utf-8",
    )
    with pytest.raises(ValidationError) as info:
        list(read_examples(broken))
    assert info.value.index == 2 and info.value.issues == ("invalid_json",)
