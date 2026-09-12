from __future__ import annotations

import pytest

from wyra.domain import Example, Message, Role
from wyra.errors import FormatError, ValidationError
from wyra.formats import (
    FORMATS,
    AlpacaFormat,
    OpenAIChatFormat,
    ShareGPTFormat,
    chat_record_issues,
    detect_format,
    get_format,
)


def test_openai_chat_round_trip(sample_examples: list[Example]) -> None:
    fmt = OpenAIChatFormat()
    for example in sample_examples:
        assert fmt.decode(fmt.encode(example)) == example


def test_openai_chat_keeps_name_and_weight_only_when_set() -> None:
    fmt = OpenAIChatFormat()
    example = Example((Message(Role.USER, "q", name="ana"), Message(Role.ASSISTANT, "a", weight=0)))
    record = fmt.encode(example)
    assert record == {
        "messages": [
            {"role": "user", "content": "q", "name": "ana"},
            {"role": "assistant", "content": "a", "weight": 0},
        ]
    }
    assert fmt.decode(record) == example


@pytest.mark.parametrize(
    ("record", "issue"),
    [
        ([1], "data_type"),
        ({"messages": "x"}, "missing_messages_list"),
        ({"messages": []}, "missing_messages_list"),
        ({"messages": ["x"]}, "data_type"),
        (
            {"messages": [{"content": "hi"}, {"role": "assistant", "content": "a"}]},
            "message_missing_key",
        ),
        (
            {
                "messages": [
                    {"role": "user", "content": "hi", "foo": 1},
                    {"role": "assistant", "content": "a"},
                ]
            },
            "message_unrecognized_key",
        ),
        (
            {
                "messages": [
                    {"role": "robot", "content": "hi"},
                    {"role": "assistant", "content": "a"},
                ]
            },
            "unrecognized_role",
        ),
        (
            {"messages": [{"role": "user", "content": ""}, {"role": "assistant", "content": "a"}]},
            "missing_content",
        ),
        ({"messages": [{"role": "user", "content": "hi"}]}, "example_missing_assistant_message"),
        (
            {
                "messages": [
                    {"role": "assistant", "content": "a"},
                    {"role": "user", "content": "hi"},
                ]
            },
            "last_not_assistant",
        ),
        (
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "system", "content": "s"},
                    {"role": "assistant", "content": "a"},
                ]
            },
            "system_not_first",
        ),
        (
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "a", "tool_calls": []},
                ]
            },
            "unsupported_tool_calls",
        ),
        (
            {"messages": [{"role": "tool", "content": "x"}, {"role": "assistant", "content": "a"}]},
            "unsupported_tool_calls",
        ),
        (
            {
                "messages": [
                    {"role": "user", "content": "hi", "weight": 1},
                    {"role": "assistant", "content": "a"},
                ]
            },
            "invalid_weight",
        ),
    ],
)
def test_chat_record_issues(record: object, issue: str) -> None:
    assert issue in chat_record_issues(record)
    with pytest.raises(ValidationError) as info:
        OpenAIChatFormat().decode(record)
    assert issue in info.value.issues


def test_valid_chat_record_has_no_issues() -> None:
    record = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
    assert chat_record_issues(record) == []


def test_alpaca_round_trip_and_limits() -> None:
    fmt = AlpacaFormat()
    example = Example.qa("Traduza.", "clean code", system="Seja breve.")
    record = fmt.encode(example)
    assert record == {
        "instruction": "Traduza.",
        "input": "",
        "output": "clean code",
        "system": "Seja breve.",
    }
    assert fmt.decode(record) == example
    assert fmt.decode({"instruction": "Traduza.", "input": "código", "output": "code"}) == (
        Example.qa("Traduza.\n\ncódigo", "code")
    )
    multi_turn = Example(
        (
            Message(Role.USER, "1"),
            Message(Role.ASSISTANT, "2"),
            Message(Role.USER, "3"),
            Message(Role.ASSISTANT, "4"),
        )
    )
    with pytest.raises(FormatError):
        fmt.encode(multi_turn)
    with pytest.raises(ValidationError):
        fmt.decode({"instruction": "x"})
    with pytest.raises(ValidationError) as info:
        fmt.decode({"instruction": "x", "output": " "})
    assert "missing_content" in info.value.issues


def test_sharegpt_round_trip_and_errors() -> None:
    fmt = ShareGPTFormat()
    example = Example(
        (
            Message(Role.SYSTEM, "s"),
            Message(Role.USER, "1"),
            Message(Role.ASSISTANT, "2"),
            Message(Role.USER, "3"),
            Message(Role.ASSISTANT, "4"),
        )
    )
    record = fmt.encode(example)
    assert record["conversations"][1] == {"from": "human", "value": "1"}
    assert fmt.decode(record) == example
    assert fmt.decode(
        {"conversations": [{"from": "user", "value": "q"}, {"from": "assistant", "value": "a"}]}
    ) == (Example.qa("q", "a"))
    with pytest.raises(ValidationError) as info:
        fmt.decode({"conversations": [{"from": "alien", "value": "q"}]})
    assert "unrecognized_role" in info.value.issues
    with pytest.raises(ValidationError):
        fmt.decode({"conversations": [{"from": "human"}]})
    with pytest.raises(ValidationError):
        fmt.decode({"nope": []})


def test_detect_and_get_format() -> None:
    assert detect_format({"messages": []}).name == "openai-chat"
    assert detect_format({"conversations": []}).name == "sharegpt"
    assert detect_format({"instruction": "i", "output": "o"}).name == "alpaca"
    with pytest.raises(FormatError):
        detect_format({"foo": 1})
    with pytest.raises(ValidationError) as info:
        detect_format([1, 2, 3])
    assert "data_type" in info.value.issues
    assert detect_format({"messages": "wrong type"}).name == "openai-chat"
    assert get_format("alpaca") is FORMATS["alpaca"]
    assert get_format(FORMATS["sharegpt"]) is FORMATS["sharegpt"]
    with pytest.raises(FormatError):
        get_format("yaml")
