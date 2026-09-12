"""Chunkers: split a Document into smaller Documents before generation.

Deterministic converters usually want the whole document (``NoChunker``). LLM-backed
generation wants chunks small enough that the model attends to all of it: context rot
degrades output well before the window limit, so the budget is a quality decision, not a
cost one.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator

from .domain import Document
from .ports import TokenCounter
from .tokens import ApproxTokenCounter

_FENCE = re.compile(r"^\s*(```|~~~)")
BLOCK_SEPARATOR = "\n\n"


class NoChunker:
    """Null object: yields the document unchanged (nothing to split)."""

    name = "none"

    def chunk(self, doc: Document) -> Iterator[Document]:
        if doc.text.strip():
            yield doc


def split_blocks(text: str) -> list[str]:
    """Split on blank lines, keeping fenced code blocks whole."""
    blocks: list[str] = []
    buffer: list[str] = []
    in_fence = False

    def flush() -> None:
        block = "\n".join(buffer).strip("\n")
        if block.strip():
            blocks.append(block)
        buffer.clear()

    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            buffer.append(line)
            continue
        if not in_fence and not line.strip():
            flush()
            continue
        buffer.append(line)
    flush()
    return blocks


class ParagraphChunker:
    """Group whole paragraphs up to a character budget, never splitting a code fence."""

    name = "paragraphs"

    def __init__(self, max_chars: int = 3000, min_chars: int = 200) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        self.max_chars = max_chars
        self.min_chars = min_chars

    def chunk(self, doc: Document) -> Iterator[Document]:
        yield from _group(doc, split_blocks(doc.text), len, self.max_chars, self.min_chars)


class WindowChunker:
    """Group whole paragraphs up to a token budget, measured by an injected counter."""

    name = "window"

    def __init__(
        self,
        max_tokens: int = 800,
        *,
        counter: TokenCounter | None = None,
        min_tokens: int = 50,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.counter = counter or ApproxTokenCounter()
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens

    def chunk(self, doc: Document) -> Iterator[Document]:
        yield from _group(
            doc, split_blocks(doc.text), self.counter.count, self.max_tokens, self.min_tokens
        )


def _group(
    doc: Document,
    blocks: list[str],
    measure: Callable[[str], int],
    maximum: int,
    minimum: int,
) -> Iterator[Document]:
    """Pack blocks into chunks, emitting a chunk when adding the next block would overflow."""
    separator = measure(BLOCK_SEPARATOR)
    current: list[str] = []
    current_size = 0
    index = 0

    def emit(parts: list[str], position: int) -> Document:
        return Document(
            text=BLOCK_SEPARATOR.join(parts),
            source=doc.source,
            sha256=doc.sha256,
            index=position,
        )

    for block in blocks:
        size = measure(block)
        if current:
            size += separator
        if current and current_size + size > maximum:
            size -= separator
            yield emit(current, index)
            index += 1
            current, current_size = [], 0
        current.append(block)
        current_size += size
        if current_size >= maximum:
            yield emit(current, index)
            index += 1
            current, current_size = [], 0

    if current and (current_size >= minimum or index == 0):
        yield emit(current, index)
