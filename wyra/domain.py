"""Domain model: the value objects every other module builds on. Stdlib only.

An ``Example`` that exists is valid by construction: invariants are enforced in
``__post_init__`` and raise ``ValidationError``. External data (JSONL on disk, model
output) goes through ``formats``/``validation`` first, which report *why* a record is bad.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from .errors import ValidationError

_WHITESPACE = re.compile(r"\s+")


class Role(str, Enum):
    """Conversation roles supported in 0.1.0. Tool/function roles are rejected on input."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

    @classmethod
    def parse(cls, value: object) -> Role:
        if isinstance(value, Role):
            return value
        if isinstance(value, str):
            try:
                return cls(value)
            except ValueError:
                pass
        raise ValidationError(f"unrecognized role: {value!r}", issues=("unrecognized_role",))


def normalize_text(text: str) -> str:
    """Canonical form used for fingerprints: NFC, whitespace collapsed, case-folded."""
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text)).strip().casefold()


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str
    name: str | None = None
    weight: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", Role.parse(self.role))
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValidationError(
                "message content must be a non-empty string", issues=("missing_content",)
            )
        if self.name is not None and (not isinstance(self.name, str) or not self.name.strip()):
            raise ValidationError(
                "message name must be a non-empty string", issues=("invalid_name",)
            )
        if self.weight is not None:
            if isinstance(self.weight, bool) or self.weight not in (0, 1):
                raise ValidationError("weight must be 0 or 1", issues=("invalid_weight",))
            if self.role is not Role.ASSISTANT:
                raise ValidationError(
                    "weight is only allowed on assistant messages",
                    issues=("invalid_weight",),
                )


@dataclass(frozen=True, slots=True)
class Example:
    """One training record: a conversation that ends with the assistant's answer."""

    messages: tuple[Message, ...]
    source: str | None = None

    def __post_init__(self) -> None:
        messages = tuple(self.messages)
        object.__setattr__(self, "messages", messages)
        if not messages:
            raise ValidationError(
                "example must contain at least one message", issues=("missing_messages_list",)
            )
        if not all(isinstance(m, Message) for m in messages):
            raise ValidationError("messages must be Message instances", issues=("data_type",))
        if not any(m.role is Role.ASSISTANT for m in messages):
            raise ValidationError(
                "example has no assistant message", issues=("example_missing_assistant_message",)
            )
        if messages[-1].role is not Role.ASSISTANT:
            raise ValidationError(
                "last message must be from the assistant", issues=("last_not_assistant",)
            )
        if any(m.role is Role.SYSTEM for m in messages[1:]):
            raise ValidationError(
                "system message must be the first message", issues=("system_not_first",)
            )

    @classmethod
    def qa(
        cls,
        question: str,
        answer: str,
        *,
        system: str | None = None,
        source: str | None = None,
    ) -> Example:
        """Build the common single-turn example, with an optional system prompt."""
        messages: list[Message] = []
        if system:
            messages.append(Message(Role.SYSTEM, system))
        messages.append(Message(Role.USER, question))
        messages.append(Message(Role.ASSISTANT, answer))
        return cls(tuple(messages), source=source)

    @property
    def system(self) -> str | None:
        first = self.messages[0]
        return first.content if first.role is Role.SYSTEM else None

    @property
    def turns(self) -> tuple[Message, ...]:
        """The conversation without its system prompt."""
        return self.messages[1:] if self.system is not None else self.messages

    def fingerprint(self, roles: Iterable[Role] | None = None) -> str:
        """Stable hash of the normalized content, optionally restricted to some roles."""
        wanted = None if roles is None else frozenset(roles)
        digest = hashlib.sha256()
        for message in self.messages:
            if wanted is not None and message.role not in wanted:
                continue
            digest.update(message.role.value.encode("utf-8"))
            digest.update(b"\x1f")
            digest.update(normalize_text(message.content).encode("utf-8"))
            digest.update(b"\x1e")
        return digest.hexdigest()

    def text_length(self) -> int:
        return sum(len(m.content) for m in self.messages)


@dataclass(frozen=True, slots=True)
class Document:
    """A unit of source text handed to a generator. Chunkers produce smaller Documents."""

    text: str
    source: str = "<string>"
    sha256: str = ""
    index: int = 0

    @property
    def ref(self) -> str:
        """Lineage reference such as ``docs/guide.md#3``."""
        return f"{self.source}#{self.index}"


@dataclass(frozen=True, slots=True)
class Completion:
    """What a CompletionProvider returns. No provider SDK type ever crosses this boundary."""

    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None
