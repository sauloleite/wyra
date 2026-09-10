from __future__ import annotations

from pathlib import Path

import pytest

from wyra.chunking import NoChunker, ParagraphChunker, WindowChunker, split_blocks
from wyra.domain import Document


def test_no_chunker_is_identity_and_drops_empty() -> None:
    chunker = NoChunker()
    doc = Document("text", source="a.md")
    assert list(chunker.chunk(doc)) == [doc]
    assert list(chunker.chunk(Document("   \n "))) == []
    assert chunker.name == "none"


def test_split_blocks_keeps_code_fences_whole() -> None:
    text = "para one\n\npara two\n\n```python\n\nx = 1\n\n```\n\nlast"
    assert split_blocks(text) == ["para one", "para two", "```python\n\nx = 1\n\n```", "last"]
    assert split_blocks("   \n\n  ") == []


def test_paragraph_chunker_respects_budget_and_numbers_chunks() -> None:
    doc = Document("\n\n".join(["a" * 90] * 5), source="doc.md", sha256="abc")
    chunks = list(ParagraphChunker(max_chars=200, min_chars=1).chunk(doc))
    assert [c.index for c in chunks] == [0, 1, 2]
    assert [c.ref for c in chunks] == ["doc.md#0", "doc.md#1", "doc.md#2"]
    assert all(c.sha256 == "abc" for c in chunks)
    assert all(len(c.text) <= 200 for c in chunks)
    assert "".join(c.text for c in chunks).count("a") == 450


def test_paragraph_chunker_never_splits_a_fence(fixtures: Path) -> None:
    doc = Document((fixtures / "clean_code.md").read_text(encoding="utf-8"), source="cc.md")
    chunks = list(ParagraphChunker(max_chars=250).chunk(doc))
    joined = [c.text for c in chunks]
    assert any("def soma" in text for text in joined)
    for text in joined:
        assert text.count("```") % 2 == 0


def test_paragraph_chunker_merges_a_short_tail_and_honours_the_budget() -> None:
    doc = Document("\n\n".join(["b" * 120, "b" * 120, "tiny"]))
    chunks = list(ParagraphChunker(max_chars=130, min_chars=100).chunk(doc))
    assert [c.text for c in chunks] == ["b" * 120, "b" * 120 + "\n\ntiny"]
    assert all(len(c.text) <= 130 for c in chunks)
    single = list(ParagraphChunker(max_chars=130, min_chars=100).chunk(Document("tiny")))
    assert [c.text for c in single] == ["tiny"]


def test_paragraph_chunker_drops_a_tail_below_the_minimum() -> None:
    doc = Document("\n\n".join(["c" * 120, "c" * 120, "d" * 40]))
    chunks = list(ParagraphChunker(max_chars=130, min_chars=100).chunk(doc))
    assert [len(c.text) for c in chunks] == [120, 120]


def test_window_chunker_uses_the_injected_counter() -> None:
    class Words:
        name = "words"

        def count(self, text: str) -> int:
            return len(text.split())

    doc = Document("\n\n".join(["one two three four"] * 4))
    chunks = list(WindowChunker(max_tokens=8, counter=Words(), min_tokens=1).chunk(doc))
    assert len(chunks) == 2
    assert all(len(c.text.split()) <= 8 for c in chunks)
    assert WindowChunker().counter.name == "approx"


@pytest.mark.parametrize("factory", [ParagraphChunker, WindowChunker])
def test_budget_must_be_positive(factory: type) -> None:
    with pytest.raises(ValueError):
        factory(0)
