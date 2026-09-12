"""Markdown sections into examples, with no model involved.

A well-written document already carries the pairs: the heading is the question and the
section body is the answer. This is the highest quality path in the library because
nothing is invented.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from typing import Any

from ..domain import Document, Example
from ..prompts import EN, Prompts

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


class MarkdownSectionGenerator:
    """Turn each ATX heading and its body into one example.

    ``min_body_chars`` skips headings that only introduce subsections. Parent headings are
    prepended to the question so a section named "Overview" is not ambiguous across files.
    """

    name = "markdown-sections"

    def __init__(
        self,
        *,
        prompts: Prompts = EN,
        min_body_chars: int = 80,
        include_parents: bool = True,
        system: str | None = None,
        max_level: int = 6,
    ) -> None:
        self.prompts = prompts
        self.min_body_chars = min_body_chars
        self.include_parents = include_parents
        self.system = system
        self.max_level = max_level

    def describe(self) -> Mapping[str, Any]:
        return {
            "lang": self.prompts.lang,
            "min_body_chars": self.min_body_chars,
            "include_parents": self.include_parents,
            "max_level": self.max_level,
            "system": self.system,
        }

    def generate(self, doc: Document) -> Iterator[Example]:
        for level, title, body in self._sections(doc.text):
            if level > self.max_level or len(body) < self.min_body_chars:
                continue
            yield Example.qa(self._question(title), body, system=self.system, source=doc.ref)

    def _question(self, title: list[str]) -> str:
        leaf = title[-1]
        if self.include_parents and len(title) > 1:
            return self.prompts.explain_section_in.format(parent=" > ".join(title[:-1]), title=leaf)
        return self.prompts.explain_section.format(title=leaf)

    @staticmethod
    def _sections(text: str) -> Iterator[tuple[int, list[str], str]]:
        """Yield ``(level, heading path, body)``. Headings inside code fences are ignored."""
        stack: list[tuple[int, str]] = []
        current: tuple[int, list[str]] | None = None
        body: list[str] = []
        in_fence = False

        for line in text.splitlines():
            if _FENCE.match(line):
                in_fence = not in_fence
            match = None if in_fence else _HEADING.match(line)
            if match is None:
                if current is not None:
                    body.append(line)
                continue
            if current is not None:
                yield current[0], current[1], "\n".join(body).strip()
            level, title = len(match.group(1)), match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            path = [t for _, t in stack] + [title]
            stack.append((level, title))
            current, body = (level, path), []

        if current is not None:
            yield current[0], current[1], "\n".join(body).strip()
