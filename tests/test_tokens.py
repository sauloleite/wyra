from __future__ import annotations

import os

import pytest

from wyra.domain import Example, Message, Role
from wyra.errors import ConfigError
from wyra.tokens import (
    ApproxTokenCounter,
    TiktokenCounter,
    count_assistant_tokens,
    count_example_tokens,
)


def test_approx_counter_is_monotonic_and_zero_on_empty() -> None:
    counter = ApproxTokenCounter()
    assert counter.name == "approx"
    assert counter.count("") == 0
    short = counter.count("Olá, mundo!")
    long = counter.count("Olá, mundo! " * 20)
    assert 0 < short < long
    assert counter.count("日本語のテキスト") >= 1


def test_cookbook_accounting() -> None:
    class OneTokenPerChar:
        name = "chars"

        def count(self, text: str) -> int:
            return len(text)

    example = Example((Message(Role.USER, "ab", name="n"), Message(Role.ASSISTANT, "abc")))
    # priming 3 + (3 + len("user") + 2 + 1 + len("n")) + (3 + len("assistant") + 3)
    assert count_example_tokens(example, OneTokenPerChar()) == 3 + (3 + 4 + 2 + 1 + 1) + (3 + 9 + 3)
    assert count_assistant_tokens(example, OneTokenPerChar()) == 3


def test_tiktoken_counter_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "tiktoken":
            raise ImportError("no tiktoken")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ConfigError, match="wyra\\[tokens\\]"):
        TiktokenCounter()


@pytest.mark.extras
@pytest.mark.skipif(
    not os.environ.get("WYRA_LIVE_TESTS"), reason="tiktoken downloads its encoding on first use"
)
def test_tiktoken_counter_live() -> None:
    counter = TiktokenCounter()
    assert counter.name == "tiktoken:cl100k_base"
    assert counter.count("hello world") == 2
