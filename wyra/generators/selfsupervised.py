"""Self-supervised generators: the text supervises itself, with no model and no labels.

These teach form and vocabulary, not conversation. Fine-tuning is for form, not for facts,
so this is exactly the shape of data it is good at absorbing. Use them for domain style;
use retrieval when the problem is "the model does not know our documents".
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterator, Mapping
from typing import Any

from ..chunking import split_blocks
from ..domain import Document, Example
from ..prompts import EN, Prompts

BLANK = "____"
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Þ0-9\"'“(])")
_FENCE_LINE = re.compile(r"^\s*(```|~~~)")

# Candidates worth hiding: numbers, quoted terms, acronyms and capitalized sequences.
_CANDIDATE = re.compile(
    r"\"[^\"]{3,40}\"|“[^”]{3,40}”|\b\d[\d.,/:%-]*\b|\b[A-ZÀ-ÖØ-Þ]{2,}\b"
    r"|\b[A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ]+(?:\s+[A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ]+)*\b"
)


def split_sentences(text: str) -> list[str]:
    """Regex sentence split. Abbreviations such as "Sr." or "etc." will split wrongly."""
    flat = " ".join(line.strip() for line in text.splitlines() if line.strip())
    return [s.strip() for s in _SENTENCE_END.split(flat) if s.strip()]


def prose_blocks(text: str) -> list[str]:
    """Paragraphs without code fences, which are not prose and must not be cut."""
    return [b for b in split_blocks(text) if not _FENCE_LINE.match(b)]


class ContinuationGenerator:
    """Ask the model to continue the text: the next sentences are the answer."""

    name = "continuation"

    def __init__(
        self,
        *,
        prompts: Prompts = EN,
        context_sentences: int = 3,
        target_sentences: int = 1,
        min_context_chars: int = 80,
        min_target_chars: int = 40,
        system: str | None = None,
    ) -> None:
        if context_sentences < 1 or target_sentences < 1:
            raise ValueError("context_sentences and target_sentences must be >= 1")
        self.prompts = prompts
        self.context_sentences = context_sentences
        self.target_sentences = target_sentences
        self.min_context_chars = min_context_chars
        self.min_target_chars = min_target_chars
        self.system = system

    def describe(self) -> Mapping[str, Any]:
        return {
            "lang": self.prompts.lang,
            "context_sentences": self.context_sentences,
            "target_sentences": self.target_sentences,
            "min_context_chars": self.min_context_chars,
            "min_target_chars": self.min_target_chars,
            "system": self.system,
        }

    def generate(self, doc: Document) -> Iterator[Example]:
        window = self.context_sentences + self.target_sentences
        for block in prose_blocks(doc.text):
            sentences = split_sentences(block)
            for start in range(0, len(sentences) - window + 1, window):
                context = " ".join(sentences[start : start + self.context_sentences])
                target = " ".join(sentences[start + self.context_sentences : start + window])
                if len(context) < self.min_context_chars or len(target) < self.min_target_chars:
                    continue
                yield Example.qa(
                    self.prompts.continue_text.format(context=context),
                    target,
                    system=self.system,
                    source=doc.ref,
                )


class ClozeGenerator:
    """Hide a term and ask for it back: cloze deletion, the oldest trick in the book."""

    name = "cloze"

    def __init__(
        self,
        *,
        prompts: Prompts = EN,
        max_per_block: int = 2,
        min_block_chars: int = 120,
        seed: int = 42,
        system: str | None = None,
    ) -> None:
        self.prompts = prompts
        self.max_per_block = max_per_block
        self.min_block_chars = min_block_chars
        self.seed = seed
        self.system = system

    def describe(self) -> Mapping[str, Any]:
        return {
            "lang": self.prompts.lang,
            "max_per_block": self.max_per_block,
            "min_block_chars": self.min_block_chars,
            "seed": self.seed,
            "system": self.system,
        }

    def generate(self, doc: Document) -> Iterator[Example]:
        rng = random.Random(f"{self.seed}:{doc.ref}")
        for block in prose_blocks(doc.text):
            if len(block) < self.min_block_chars:
                continue
            spans = _candidates(block)
            if not spans:
                continue
            rng.shuffle(spans)
            for start, end in sorted(spans[: self.max_per_block]):
                answer = block[start:end]
                masked = f"{block[:start]}{BLANK}{block[end:]}"
                yield Example.qa(
                    self.prompts.fill_blank.format(text=masked),
                    answer,
                    system=self.system,
                    source=doc.ref,
                )


def _candidates(block: str) -> list[tuple[int, int]]:
    """Spans worth hiding: not at the very start, long enough to be a real answer."""
    spans = []
    for match in _CANDIDATE.finditer(block):
        start, end = match.span()
        if start == 0 or end - start < 3:
            continue
        spans.append((start, end))
    return spans
