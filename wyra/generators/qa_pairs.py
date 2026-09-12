"""Question/answer pairs that already exist in the text, extracted with no model.

Two shapes are recognized: explicit markers ("Q:"/"A:", "Pergunta:"/"Resposta:") and
rhetorical questions in prose followed by the paragraph that answers them.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from ..chunking import split_blocks
from ..domain import Document, Example

DEFAULT_MARKERS: tuple[tuple[str, str], ...] = (
    ("q", "a"),
    ("question", "answer"),
    ("pergunta", "resposta"),
    ("p", "r"),
)

_DECORATION = re.compile(r"^\s*(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>\s+)+|[*_`]+")


class QAPairExtractor:
    """Extract pairs the document already contains.

    Markers are matched line by line, so ``Pergunta:`` and ``Resposta:`` work whether they
    sit in the same paragraph or in separate ones. When no marker is found,
    ``prose_questions`` picks up paragraphs ending in a question mark and uses the next
    paragraph as the answer, which is how most articles are written.
    """

    name = "qa-pairs"

    def __init__(
        self,
        *,
        markers: Sequence[tuple[str, str]] = DEFAULT_MARKERS,
        prose_questions: bool = True,
        min_answer_chars: int = 20,
        max_answer_blocks: int = 2,
        system: str | None = None,
    ) -> None:
        if max_answer_blocks < 1:
            raise ValueError("max_answer_blocks must be >= 1")
        self.markers = tuple(markers)
        self.prose_questions = prose_questions
        self.min_answer_chars = min_answer_chars
        self.max_answer_blocks = max_answer_blocks
        self.system = system
        self._question_re, self._answer_re = _marker_patterns(self.markers)

    def describe(self) -> Mapping[str, Any]:
        return {
            "markers": [list(pair) for pair in self.markers],
            "prose_questions": self.prose_questions,
            "min_answer_chars": self.min_answer_chars,
            "max_answer_blocks": self.max_answer_blocks,
            "system": self.system,
        }

    def generate(self, doc: Document) -> Iterator[Example]:
        marked = list(self._from_markers(doc))
        if marked:
            yield from marked
            return
        if self.prose_questions:
            yield from self._from_prose(doc)

    def _from_markers(self, doc: Document) -> Iterator[Example]:
        question: str | None = None
        blocks: list[list[str]] = []
        open_field: str | None = None

        for line in doc.text.splitlines():
            stripped = line.strip()
            starts_question = self._question_re.match(line)
            starts_answer = self._answer_re.match(line)

            if starts_question:
                yield from self._emit(question, blocks, doc)
                question = starts_question.group("text").strip()
                blocks, open_field = [], "question"
            elif starts_answer and question is not None:
                blocks, open_field = [[starts_answer.group("text").strip()]], "answer"
            elif not stripped:
                if open_field == "answer" and len(blocks) < self.max_answer_blocks:
                    blocks.append([])
                else:
                    open_field = None
            elif open_field == "question" and question is not None:
                question = f"{question} {stripped}"
            elif open_field == "answer":
                blocks[-1].append(stripped)

        yield from self._emit(question, blocks, doc)

    def _from_prose(self, doc: Document) -> Iterator[Example]:
        blocks = split_blocks(doc.text)
        for index, block in enumerate(blocks):
            question = _trailing_question(block)
            if question is None or index + 1 >= len(blocks):
                continue
            answer = blocks[index + 1].strip()
            if _trailing_question(answer) is not None:
                continue
            yield from self._emit(question, [[answer]], doc)

    def _emit(
        self, question: str | None, blocks: Sequence[Sequence[str]], doc: Document
    ) -> Iterator[Example]:
        if not question:
            return
        paragraphs = [" ".join(lines).strip() for lines in blocks]
        answer = "\n\n".join(p for p in paragraphs if p).strip()
        if len(answer) < self.min_answer_chars:
            return
        yield Example.qa(question, answer, system=self.system, source=doc.ref)


def _marker_patterns(markers: Sequence[tuple[str, str]]) -> tuple[re.Pattern[str], re.Pattern[str]]:
    questions = "|".join(re.escape(q) for q, _ in markers)
    answers = "|".join(re.escape(a) for _, a in markers)
    return (
        re.compile(rf"^\s*(?:{questions})\s*[:.\-]\s*(?P<text>.+)$", re.IGNORECASE),
        re.compile(rf"^\s*(?:{answers})\s*[:.\-]\s*(?P<text>.+)$", re.IGNORECASE),
    )


def _trailing_question(block: str) -> str | None:
    """Return the question a block ends with, if any, stripped of Markdown decoration."""
    lines = [line.strip() for line in block.strip().splitlines() if line.strip()]
    if not lines or not lines[-1].endswith("?"):
        return None
    question = _DECORATION.sub("", lines[-1]).strip().rstrip("*_`").strip()
    return question or None
