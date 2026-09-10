"""Token counting behind a port: an approximate stdlib counter and an optional tiktoken one.

``count_example_tokens`` follows the OpenAI cookbook accounting (3 tokens per message,
1 per name, 3 for the assistant priming) so budgets match what the API will see.
"""

from __future__ import annotations

import re

from .domain import Example, Role
from .errors import ConfigError
from .ports import TokenCounter

_TOKENISH = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class ApproxTokenCounter:
    """Dependency-free estimate: roughly one token per 4 characters or per 0.75 words."""

    name = "approx"

    def count(self, text: str) -> int:
        if not text:
            return 0
        words = len(_TOKENISH.findall(text))
        return max(1, round(max(words * 1.3, len(text) / 4)))


class TiktokenCounter:
    """Exact counts for OpenAI models. Needs ``pip install 'wyra[tokens]'``.

    tiktoken downloads the encoding file on first use, so keep this out of unit tests.
    """

    name: str

    def __init__(self, encoding: str = "cl100k_base") -> None:
        try:
            import tiktoken
        except ImportError as exc:
            raise ConfigError(
                "TiktokenCounter needs the optional dependency: pip install 'wyra[tokens]'"
            ) from exc
        self._encoding = tiktoken.get_encoding(encoding)
        self.name = f"tiktoken:{encoding}"

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text))


def count_example_tokens(
    example: Example,
    counter: TokenCounter,
    *,
    tokens_per_message: int = 3,
    tokens_per_name: int = 1,
    priming: int = 3,
) -> int:
    total = priming
    for message in example.messages:
        total += tokens_per_message
        total += counter.count(message.role.value) + counter.count(message.content)
        if message.name is not None:
            total += tokens_per_name + counter.count(message.name)
    return total


def count_assistant_tokens(example: Example, counter: TokenCounter) -> int:
    return sum(counter.count(m.content) for m in example.messages if m.role is Role.ASSISTANT)
