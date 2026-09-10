from __future__ import annotations

import pytest

from wyra.domain import Completion, Document, Example, Message, Role, normalize_text
from wyra.errors import ValidationError


def test_message_accepts_role_strings() -> None:
    message = Message("user", "hi")  # type: ignore[arg-type]
    assert message.role is Role.USER


@pytest.mark.parametrize(
    ("kwargs", "issue"),
    [
        ({"role": "robot", "content": "hi"}, "unrecognized_role"),
        ({"role": Role.USER, "content": "   "}, "missing_content"),
        ({"role": Role.USER, "content": 42}, "missing_content"),
        ({"role": Role.ASSISTANT, "content": "ok", "weight": 2}, "invalid_weight"),
        ({"role": Role.ASSISTANT, "content": "ok", "weight": True}, "invalid_weight"),
        ({"role": Role.USER, "content": "ok", "weight": 1}, "invalid_weight"),
        ({"role": Role.USER, "content": "ok", "name": ""}, "invalid_name"),
    ],
)
def test_message_invariants(kwargs: dict[str, object], issue: str) -> None:
    with pytest.raises(ValidationError) as info:
        Message(**kwargs)  # type: ignore[arg-type]
    assert issue in info.value.issues


def test_example_invariants() -> None:
    user = Message(Role.USER, "hi")
    assistant = Message(Role.ASSISTANT, "hello")
    system = Message(Role.SYSTEM, "be nice")

    with pytest.raises(ValidationError) as info:
        Example(())
    assert "missing_messages_list" in info.value.issues

    with pytest.raises(ValidationError) as info:
        Example((user,))
    assert "example_missing_assistant_message" in info.value.issues

    with pytest.raises(ValidationError) as info:
        Example((user, assistant, user))
    assert "last_not_assistant" in info.value.issues

    with pytest.raises(ValidationError) as info:
        Example((user, system, assistant))
    assert "system_not_first" in info.value.issues

    with pytest.raises(ValidationError) as info:
        Example((user, "assistant"))  # type: ignore[arg-type]
    assert "data_type" in info.value.issues

    example = Example([system, user, assistant])  # type: ignore[arg-type]
    assert isinstance(example.messages, tuple)
    assert example.system == "be nice"
    assert example.turns == (user, assistant)


def test_qa_builder_and_source() -> None:
    example = Example.qa("q?", "a.", system="sys", source="doc.md#1")
    assert [m.role for m in example.messages] == [Role.SYSTEM, Role.USER, Role.ASSISTANT]
    assert example.source == "doc.md#1"
    assert Example.qa("q?", "a.").system is None
    assert example.text_length() == len("sys") + len("q?") + len("a.")


def test_fingerprint_ignores_whitespace_case_and_unicode_form() -> None:
    a = Example.qa("O que é  DRY?", "Não repita.")
    b = Example.qa("o que é dry?", "não repita.  ")
    c = Example.qa("O que é DRY?", "Repita.")
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()
    assert a.fingerprint(roles=[Role.USER]) == c.fingerprint(roles=[Role.USER])
    assert normalize_text("  Ã  b ") == "ã b"


def test_examples_are_hashable_value_objects() -> None:
    assert Example.qa("q", "a") == Example.qa("q", "a")
    assert len({Example.qa("q", "a"), Example.qa("q", "a")}) == 1
    with pytest.raises(AttributeError):
        Example.qa("q", "a").source = "x"  # type: ignore[misc]


def test_document_ref_and_completion_defaults() -> None:
    assert Document("text", source="a.md", index=3).ref == "a.md#3"
    assert Document("text").source == "<string>"
    completion = Completion("{}", model="fake")
    assert completion.input_tokens is None and completion.finish_reason is None
