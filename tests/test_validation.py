from __future__ import annotations

from pathlib import Path

from wyra.domain import Example
from wyra.errors import ValidationError
from wyra.tokens import ApproxTokenCounter
from wyra.validation import validate_jsonl, validate_records


def test_validate_good_file(fixtures: Path) -> None:
    report = validate_jsonl(fixtures / "good.jsonl")
    assert report.ok
    assert (report.total, report.valid) == (3, 3)
    assert report.format == "openai-chat"
    assert report.issues == {}
    assert report.tokens is not None
    assert report.tokens.count == 3 and report.tokens.min > 0
    assert report.tokens.over_budget == 0
    assert "3/3 valid" in report.summary()


def test_validate_bad_file_counts_every_issue(fixtures: Path) -> None:
    report = validate_jsonl(fixtures / "bad.jsonl")
    assert not report.ok
    assert report.total == 11 and report.valid == 1
    for issue in (
        "data_type",
        "missing_messages_list",
        "message_unrecognized_key",
        "unrecognized_role",
        "missing_content",
        "example_missing_assistant_message",
        "last_not_assistant",
        "unsupported_tool_calls",
        "invalid_json",
        "unknown_format",
    ):
        assert report.issues[issue] >= 1, issue
    lines = [line for line, _ in report.invalid]
    assert lines == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert "invalid_json: 1" in report.summary()


def test_validate_records_detects_format_and_limits_listing() -> None:
    records = [(i, {"instruction": "i", "output": "o"}) for i in range(1, 4)]
    records += [(9, {"instruction": "i", "output": ""}), (10, {"weird": 1})]
    report = validate_records(records, max_listed=1, counter=ApproxTokenCounter())
    assert report.format == "alpaca"
    assert report.valid == 3 and report.total == 5
    # once the format is pinned by the first record, later records are checked against it
    assert report.issues == {"missing_content": 1, "missing_messages_list": 1}
    assert report.invalid == [(9, ("missing_content",))]
    assert report.tokens is not None and report.tokens.count == 3


def test_validate_records_with_explicit_format_and_error_objects() -> None:
    bad = ValidationError("broken", issues=("invalid_json",), index=2)
    report = validate_records([(2, bad)], "sharegpt")
    assert report.format == "sharegpt"
    assert report.issues == {"invalid_json": 1}
    assert not report.ok
    assert report.tokens is None


def test_empty_report_is_not_ok() -> None:
    report = validate_records([], "openai-chat")
    assert not report.ok
    assert "<records>: 0/0" in report.summary()


def test_add_valid_accumulates_tokens() -> None:
    report = validate_records(
        [
            (
                1,
                {
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "yo"},
                    ]
                },
            )
        ],
        counter=ApproxTokenCounter(),
        budget=1,
    )
    assert report.tokens is not None
    assert report.tokens.over_budget == 1
    assert report.tokens.as_dict()["tokens"]["over_budget"] == 1
    assert isinstance(Example.qa("a", "b"), Example)
