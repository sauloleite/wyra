"""Ports: the interfaces the core defines and the edges implement.

Everything here is a ``typing.Protocol``. The only abstract base class in the package is
``generators.llm.LLMExampleGenerator``, because it carries shared implementation.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .domain import Completion, Document, Example, Message


@runtime_checkable
class CompletionProvider(Protocol):
    """A model that turns a conversation into one completion (Ollama, OpenAI, Gemini, fake)."""

    name: str
    model: str

    def complete(
        self,
        messages: Sequence[Message],
        *,
        json_schema: Mapping[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> Completion: ...


@runtime_checkable
class ExampleGenerator(Protocol):
    """Turns one Document into zero or more Examples."""

    name: str

    def describe(self) -> Mapping[str, Any]:
        """Parameters worth recording in the dataset manifest."""
        ...

    def generate(self, doc: Document) -> Iterable[Example]: ...


class Chunker(Protocol):
    name: str

    def chunk(self, doc: Document) -> Iterable[Document]: ...


class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int: ...


class CurationStep(Protocol):
    """A filter or transform over the example stream. May keep state (dedup)."""

    name: str

    def apply(self, examples: Iterable[Example]) -> Iterator[Example]: ...


class DatasetWriter(Protocol):
    format: str

    def write(self, examples: Iterable[Example], destination: str | Path) -> int:
        """Write every example and return how many were written."""
        ...
