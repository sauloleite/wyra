"""LLM-backed generation: the only path that invents new question/answer pairs.

The skeleton below is shared implementation (prompt assembly, structured-output request,
JSON extraction, schema check, grounding filter, one retry), which is why this is an
abstract class and not a Protocol: an interface cannot carry an algorithm. Subclasses
supply the three things that must agree with each other, namely the task prompt, the
response schema and how a payload maps to examples.

Model output is data, never instruction: nothing it returns changes control flow, and
everything it returns is validated by code before reaching the dataset.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from ..domain import Document, Example, Message, Role
from ..errors import GenerationError, TruncatedOutputError
from ..ports import CompletionProvider, TokenCounter
from ..prompts import EN, Prompts
from ..tokens import ApproxTokenCounter

logger = logging.getLogger("wyra")

PROMPT_VERSION = "1"
_FENCE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*|\s*```\s*$")
_WORD = re.compile(r"\w{4,}", re.UNICODE)


def extract_json(text: str) -> Any:
    """Parse the JSON in a model reply, tolerating code fences and surrounding prose.

    Deliberately not a JSON repairer: silently fixing malformed output hides the failure
    that the retry is there to surface.
    """
    stripped = _FENCE.sub("", text.strip())
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    for opening, closing in (("{", "}"), ("[", "]")):
        start = stripped.find(opening)
        end = stripped.rfind(closing)
        if start != -1 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise GenerationError(f"model reply is not JSON: {text[:200]!r}")


class LLMExampleGenerator(ABC):
    """Template Method: fixed pipeline, three abstract hooks."""

    name = "llm"

    def __init__(
        self,
        provider: CompletionProvider,
        *,
        prompts: Prompts = EN,
        per_chunk: int = 3,
        temperature: float = 0.2,
        max_retries: int = 1,
        min_grounding: float = 0.2,
        max_input_tokens: int = 6000,
        counter: TokenCounter | None = None,
        system: str | None = None,
    ) -> None:
        self.provider = provider
        self.prompts = prompts
        self.per_chunk = per_chunk
        self.temperature = temperature
        self.max_retries = max_retries
        self.min_grounding = min_grounding
        self.max_input_tokens = max_input_tokens
        self.counter = counter or ApproxTokenCounter()
        self.system = system

    # --- hooks -----------------------------------------------------------------

    @abstractmethod
    def task_prompt(self, doc: Document, count: int) -> str:
        """The user message asking for ``count`` examples from this chunk."""

    @abstractmethod
    def response_schema(self, count: int) -> Mapping[str, Any]:
        """A flat JSON schema every provider dialect accepts, bounded to ``count`` items.

        The bound is the point. Constrained decoding follows the schema, not the prose, so
        an array without ``maxItems`` lets a weak model emit items until it runs out of
        output budget and never closes the bracket. Asking for three in the prompt is a
        request; ``maxItems`` is a guarantee.
        """

    @abstractmethod
    def parse(self, payload: Any, doc: Document) -> Iterator[Example]:
        """Map the decoded payload to examples. Raise ``GenerationError`` on a bad shape."""

    # --- skeleton --------------------------------------------------------------

    def describe(self) -> Mapping[str, Any]:
        described: dict[str, Any] = {
            "provider": self.provider.name,
            "model": self.provider.model,
            "lang": self.prompts.lang,
            "per_chunk": self.per_chunk,
            "temperature": self.temperature,
            "max_retries": self.max_retries,
            "min_grounding": self.min_grounding,
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": self.prompt_fingerprint(),
            "system": self.system,
        }
        weights = _provider_lineage(self.provider)
        if weights:
            described["weights"] = weights
        return described

    def prompt_fingerprint(self) -> str:
        """Hash of the exact instructions used, so a manifest pins the prompt version."""
        material = (
            f"{PROMPT_VERSION}|{self.prompts.qa_system}|{self.task_prompt(_SAMPLE, self.per_chunk)}"
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def generate(self, doc: Document) -> Iterator[Example]:
        if not doc.text.strip():
            return
        self._guard_size(doc)
        payload = self._ask_for(doc, self.per_chunk)
        for example in self.parse(payload, doc):
            if self._grounded(example, doc):
                yield example

    def _ask_for(self, doc: Document, wanted: int) -> Any:
        """Ask for ``wanted`` items, halving the request when the model runs out of room.

        A truncated reply used to lose the whole chunk, which on a local model with a modest
        output budget was the common case rather than the exception.
        """
        while True:
            try:
                return self._ask(doc, wanted)
            except TruncatedOutputError:
                if wanted <= 1:
                    raise
                wanted = max(1, wanted // 2)
                logger.info(
                    "%s truncated its output; asking for %d item(s) instead",
                    self.provider.model,
                    wanted,
                )

    def _ask(self, doc: Document, wanted: int) -> Any:
        """Call the provider, retrying once when the reply is not usable JSON."""
        messages = [
            Message(Role.SYSTEM, self.prompts.qa_system),
            Message(Role.USER, self.task_prompt(doc, wanted)),
        ]
        attempt = 0
        while True:
            completion = self.provider.complete(
                messages,
                json_schema=self.response_schema(wanted),
                temperature=self.temperature,
            )
            try:
                return extract_json(completion.text)
            except GenerationError:
                if completion.finish_reason == "length":
                    raise TruncatedOutputError(
                        "the model ran out of room before producing usable JSON; small "
                        "local models degenerate on long inputs, so reduce the chunker's "
                        "max_chars or use a stronger model"
                    ) from None
                if attempt >= self.max_retries:
                    raise
                attempt += 1
                messages = [
                    *messages,
                    Message(Role.ASSISTANT, completion.text or "(empty)"),
                    Message(Role.USER, self.prompts.json_only),
                ]

    def _guard_size(self, doc: Document) -> None:
        tokens = self.counter.count(doc.text)
        if tokens > self.max_input_tokens:
            raise GenerationError(
                f"chunk {doc.ref} is {tokens} tokens, above max_input_tokens="
                f"{self.max_input_tokens}; use a chunker"
            )

    def _grounded(self, example: Example, doc: Document) -> bool:
        """Keep answers that share vocabulary with the source. A blunt but cheap check."""
        if self.min_grounding <= 0:
            return True
        source = _terms(doc.text)
        answer = _terms(" ".join(m.content for m in example.messages if m.role is Role.ASSISTANT))
        if not answer or not source:
            return True
        return len(answer & source) / len(answer) >= self.min_grounding


def _provider_lineage(provider: CompletionProvider) -> Mapping[str, Any]:
    """Optional provenance from a provider. Most have none, and that must never fail a build."""
    hook = getattr(provider, "lineage", None)
    if not callable(hook):
        return {}
    try:
        data = hook()
    except Exception:  # provenance is a nicety; losing it must not lose the dataset
        return {}
    return dict(data) if isinstance(data, Mapping) else {}


def _terms(text: str) -> set[str]:
    return set(_WORD.findall(unicodedata.normalize("NFC", text).casefold()))


_SAMPLE = Document("sample", source="<fingerprint>")


class QAPairGenerator(LLMExampleGenerator):
    """Ask for question/answer pairs grounded in the chunk."""

    name = "llm-qa"

    def task_prompt(self, doc: Document, count: int) -> str:
        return self.prompts.qa_task.format(n=count, text=doc.text)

    def response_schema(self, count: int) -> Mapping[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pairs": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": max(count, 1),
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "answer": {"type": "string"},
                        },
                        "required": ["question", "answer"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["pairs"],
            "additionalProperties": False,
        }

    def parse(self, payload: Any, doc: Document) -> Iterator[Example]:
        for item in _items(payload, "pairs"):
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            if question and answer:
                yield Example.qa(question, answer, system=self.system, source=doc.ref)


class InstructionGenerator(LLMExampleGenerator):
    """Ask for instruction/input/output items, the Alpaca shape."""

    name = "llm-instruction"

    def task_prompt(self, doc: Document, count: int) -> str:
        return self.prompts.instruction_task.format(n=count, text=doc.text)

    def response_schema(self, count: int) -> Mapping[str, Any]:
        return {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": max(count, 1),
                    "items": {
                        "type": "object",
                        "properties": {
                            "instruction": {"type": "string"},
                            "input": {"type": "string"},
                            "output": {"type": "string"},
                        },
                        "required": ["instruction", "input", "output"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["items"],
            "additionalProperties": False,
        }

    def parse(self, payload: Any, doc: Document) -> Iterator[Example]:
        for item in _items(payload, "items"):
            instruction = str(item.get("instruction", "")).strip()
            output = str(item.get("output", "")).strip()
            extra = str(item.get("input", "") or "").strip()
            if not instruction or not output:
                continue
            question = f"{instruction}\n\n{extra}" if extra else instruction
            yield Example.qa(question, output, system=self.system, source=doc.ref)


def _items(payload: Any, key: str) -> Sequence[Mapping[str, Any]]:
    """Accept the documented shape, and a bare list, which small models often return."""
    if isinstance(payload, Mapping):
        payload = payload.get(key, payload.get("data"))
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise GenerationError(f"expected a list under {key!r}, got {type(payload).__name__}")
    items = [item for item in payload if isinstance(item, Mapping)]
    if not items:
        raise GenerationError(f"no usable items under {key!r}")
    return items
