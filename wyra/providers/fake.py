"""A fake completion provider: canned replies plus a record of every call.

Shipped with the library, not hidden in the test folder, because anyone building on wyra
needs to test their own generators without credentials or a network.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..domain import Completion, Message
from ..errors import ProviderError


@dataclass(frozen=True, slots=True)
class RecordedCall:
    messages: tuple[Message, ...]
    json_schema: Mapping[str, Any] | None
    temperature: float

    @property
    def prompt(self) -> str:
        return "\n\n".join(m.content for m in self.messages)


@dataclass
class FakeCompletionProvider:
    """Replies with each canned response in turn; the last one repeats when exhausted.

    A response may also be a callable taking the messages, or an exception instance to
    raise, which is how failure paths get tested.
    """

    responses: Sequence[Any] = ("{}",)
    model: str = "fake-model"
    name: str = "fake"
    finish_reason: str | None = None
    calls: list[RecordedCall] = field(default_factory=list)

    def complete(
        self,
        messages: Sequence[Message],
        *,
        json_schema: Mapping[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> Completion:
        self.calls.append(RecordedCall(tuple(messages), json_schema, temperature))
        if not self.responses:
            raise ProviderError("FakeCompletionProvider has no responses configured")
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        reply = self.responses[index]
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, Completion):
            return reply
        if isinstance(reply, Callable):  # type: ignore[arg-type]
            reply = reply(messages)
        return Completion(
            text=str(reply),
            model=self.model,
            input_tokens=sum(len(m.content) for m in messages) // 4,
            output_tokens=len(str(reply)) // 4,
            finish_reason=self.finish_reason,
        )
