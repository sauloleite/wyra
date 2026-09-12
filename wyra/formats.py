"""Dataset formats: how an Example is encoded to, and decoded from, one JSONL record.

``openai-chat`` is the canonical format (also used by Azure OpenAI, Hugging Face TRL,
Unsloth and Axolotl). ``alpaca`` and ``sharegpt`` are provided for the rest of the
ecosystem. The OpenAI cookbook validation rules live here because they *are* the schema.
"""

from __future__ import annotations

from typing import Any, Protocol

from .domain import Example, Message, Role
from .errors import FormatError, ValidationError

RECOGNIZED_MESSAGE_KEYS = frozenset(
    {"role", "content", "name", "weight", "function_call", "tool_calls", "tool_call_id", "refusal"}
)
KNOWN_ROLES = frozenset({"system", "user", "assistant", "function", "tool"})
_TOOL_KEYS = frozenset({"function_call", "tool_calls", "tool_call_id"})


class DatasetFormat(Protocol):
    name: str

    def encode(self, example: Example) -> dict[str, Any]: ...

    def decode(self, record: Any) -> Example: ...

    def matches(self, record: Any) -> bool: ...


def chat_record_issues(record: Any) -> list[str]:
    """OpenAI cookbook format checks, as issue tags. An empty list means the record is fine."""
    if not isinstance(record, dict):
        return ["data_type"]
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return ["missing_messages_list"]

    issues: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            issues.append("data_type")
            continue
        if "role" not in message or "content" not in message:
            issues.append("message_missing_key")
        if any(key not in RECOGNIZED_MESSAGE_KEYS for key in message):
            issues.append("message_unrecognized_key")
        role = message.get("role")
        if role not in KNOWN_ROLES:
            issues.append("unrecognized_role")
        elif role in ("function", "tool") or any(key in message for key in _TOOL_KEYS):
            issues.append("unsupported_tool_calls")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            issues.append("missing_content")
        weight = message.get("weight")
        if weight is not None and (
            isinstance(weight, bool) or weight not in (0, 1) or role != "assistant"
        ):
            issues.append("invalid_weight")

    dict_messages = [m for m in messages if isinstance(m, dict)]
    if not any(m.get("role") == "assistant" for m in dict_messages):
        issues.append("example_missing_assistant_message")
    elif not (isinstance(messages[-1], dict) and messages[-1].get("role") == "assistant"):
        issues.append("last_not_assistant")
    if any(m.get("role") == "system" for m in dict_messages[1:]):
        issues.append("system_not_first")
    return issues


def _encode_message(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role.value, "content": message.content}
    if message.name is not None:
        payload["name"] = message.name
    if message.weight is not None:
        payload["weight"] = message.weight
    return payload


class OpenAIChatFormat:
    name = "openai-chat"

    def encode(self, example: Example) -> dict[str, Any]:
        return {"messages": [_encode_message(m) for m in example.messages]}

    def decode(self, record: Any) -> Example:
        issues = chat_record_issues(record)
        if issues:
            raise ValidationError(
                f"invalid {self.name} record: {', '.join(sorted(set(issues)))}",
                issues=tuple(issues),
            )
        messages = tuple(
            Message(Role(m["role"]), m["content"], name=m.get("name"), weight=m.get("weight"))
            for m in record["messages"]
        )
        return Example(messages)

    def matches(self, record: Any) -> bool:
        return isinstance(record, dict) and "messages" in record


class AlpacaFormat:
    """``instruction`` / ``input`` / ``output`` records; single exchange only."""

    name = "alpaca"

    def encode(self, example: Example) -> dict[str, Any]:
        turns = example.turns
        if len(turns) != 2 or turns[0].role is not Role.USER:
            raise FormatError(
                "alpaca can only represent one user/assistant exchange "
                "(plus an optional system prompt)"
            )
        record: dict[str, Any] = {
            "instruction": turns[0].content,
            "input": "",
            "output": turns[1].content,
        }
        if example.system is not None:
            record["system"] = example.system
        return record

    def decode(self, record: Any) -> Example:
        if not self.matches(record):
            raise ValidationError(
                "alpaca record needs 'instruction' and 'output' strings",
                issues=("missing_messages_list",),
            )
        question = record["instruction"]
        extra = record.get("input")
        if isinstance(extra, str) and extra.strip():
            question = f"{question}\n\n{extra}"
        system = record.get("system")
        if not (isinstance(system, str) and system.strip()):
            system = None
        return Example.qa(question, record["output"], system=system)

    def matches(self, record: Any) -> bool:
        return isinstance(record, dict) and "instruction" in record and "output" in record


_SHAREGPT_TO_ROLE = {
    "system": Role.SYSTEM,
    "human": Role.USER,
    "user": Role.USER,
    "gpt": Role.ASSISTANT,
    "assistant": Role.ASSISTANT,
}
_ROLE_TO_SHAREGPT = {Role.SYSTEM: "system", Role.USER: "human", Role.ASSISTANT: "gpt"}


class ShareGPTFormat:
    """``conversations`` list with ``from``/``value`` turns."""

    name = "sharegpt"

    def encode(self, example: Example) -> dict[str, Any]:
        return {
            "conversations": [
                {"from": _ROLE_TO_SHAREGPT[m.role], "value": m.content} for m in example.messages
            ]
        }

    def decode(self, record: Any) -> Example:
        if not self.matches(record):
            raise ValidationError(
                "sharegpt record needs a 'conversations' list", issues=("missing_messages_list",)
            )
        messages: list[Message] = []
        for turn in record["conversations"]:
            if not isinstance(turn, dict) or not isinstance(turn.get("value"), str):
                raise ValidationError(
                    "sharegpt turn needs a string 'value'", issues=("missing_content",)
                )
            speaker = turn.get("from")
            role = _SHAREGPT_TO_ROLE.get(speaker) if isinstance(speaker, str) else None
            if role is None:
                raise ValidationError(
                    f"unrecognized sharegpt speaker: {speaker!r}", issues=("unrecognized_role",)
                )
            messages.append(Message(role, turn["value"]))
        return Example(tuple(messages))

    def matches(self, record: Any) -> bool:
        return isinstance(record, dict) and "conversations" in record


FORMATS: dict[str, DatasetFormat] = {
    fmt.name: fmt for fmt in (OpenAIChatFormat(), ShareGPTFormat(), AlpacaFormat())
}


def get_format(name: str | DatasetFormat) -> DatasetFormat:
    if not isinstance(name, str):
        return name
    try:
        return FORMATS[name]
    except KeyError:
        raise FormatError(f"unknown format {name!r}; known: {', '.join(FORMATS)}") from None


def detect_format(record: Any) -> DatasetFormat:
    """Pick the format a record belongs to.

    A record that is not a JSON object is a bad record, not an unknown format, so it
    raises ``ValidationError`` with the cookbook's ``data_type`` tag.
    """
    if not isinstance(record, dict):
        raise ValidationError("record must be a JSON object", issues=("data_type",))
    for fmt in FORMATS.values():
        if fmt.matches(record):
            return fmt
    raise FormatError(
        "could not detect the record format "
        "(expected 'messages', 'conversations' or 'instruction'/'output' keys)"
    )
