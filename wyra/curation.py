"""Curation: deterministic code between generation and disk.

Every step is a small object with a ``name`` and an ``apply`` that transforms a stream of
examples. ``run_pipeline`` wraps each step in a counter so the manifest can report how
many examples entered and left it. No model is involved: this is the structural guardrail.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from .domain import Example, Message, Role
from .ports import CurationStep, TokenCounter
from .tokens import ApproxTokenCounter, count_example_tokens

_TRAILING = re.compile(r"[ \t]+(\n|$)")
_BLANKS = re.compile(r"\n{3,}")
_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass(slots=True)
class StepCount:
    step: str
    received: int = 0
    kept: int = 0

    @property
    def dropped(self) -> int:
        return self.received - self.kept

    def as_dict(self) -> dict[str, int | str]:
        return {"step": self.step, "in": self.received, "out": self.kept}


@dataclass(slots=True)
class CurationReport:
    steps: list[StepCount] = field(default_factory=list)

    @property
    def generated(self) -> int:
        return self.steps[0].received if self.steps else 0

    @property
    def kept(self) -> int:
        return self.steps[-1].kept if self.steps else 0

    def as_list(self) -> list[dict[str, int | str]]:
        return [step.as_dict() for step in self.steps]

    def summary(self) -> str:
        return " -> ".join(f"{s.step}: {s.received}/{s.kept}" for s in self.steps)


class Normalize:
    """Unicode NFC, CRLF to LF, no trailing spaces, at most one blank line in a row.

    Leading whitespace is left alone on purpose: it carries meaning in Markdown lists and
    indented code, which answers frequently contain.
    """

    name = "normalize"

    def apply(self, examples: Iterable[Example]) -> Iterator[Example]:
        for example in examples:
            messages = tuple(
                Message(m.role, _clean(m.content), name=m.name, weight=m.weight)
                for m in example.messages
            )
            yield example if messages == example.messages else Example(messages, example.source)


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    return _BLANKS.sub("\n\n", _TRAILING.sub(r"\1", text)).strip()


class Dedup:
    """Drop repeated examples, keeping the first occurrence.

    ``key="all"`` compares the whole conversation; ``key="user"`` compares only the user
    turns, which catches the same question answered twice with different wording.
    """

    name = "dedup"

    def __init__(self, key: str = "all") -> None:
        if key not in ("all", "user"):
            raise ValueError("key must be 'all' or 'user'")
        self.key = key

    def apply(self, examples: Iterable[Example]) -> Iterator[Example]:
        roles = None if self.key == "all" else (Role.USER,)
        seen: set[str] = set()
        for example in examples:
            fingerprint = example.fingerprint(roles)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            yield example


class MinLength:
    """Drop examples whose assistant answer is too short to teach anything."""

    name = "min_length"

    def __init__(self, min_chars: int = 20, *, role: Role = Role.ASSISTANT) -> None:
        self.min_chars = min_chars
        self.role = role

    def apply(self, examples: Iterable[Example]) -> Iterator[Example]:
        for example in examples:
            content = sum(len(m.content) for m in example.messages if m.role is self.role)
            if content >= self.min_chars:
                yield example


class TokenBudget:
    """Drop examples above the per-example token limit of the target API."""

    name = "token_budget"

    def __init__(self, max_tokens: int = 16385, *, counter: TokenCounter | None = None) -> None:
        self.max_tokens = max_tokens
        self.counter = counter or ApproxTokenCounter()

    def apply(self, examples: Iterable[Example]) -> Iterator[Example]:
        for example in examples:
            if count_example_tokens(example, self.counter) <= self.max_tokens:
                yield example


class NearDuplicate:
    """Drop near-duplicates using MinHash over word shingles, banded for speed.

    Exact dedup misses "same content, one word changed"; near-duplicate examples make the
    model overfit those patterns. Pure stdlib: hashing is blake2b with a per-permutation
    salt, so results are reproducible across runs and processes.
    """

    name = "near_duplicate"

    def __init__(
        self,
        threshold: float = 0.9,
        *,
        shingle: int = 5,
        num_perm: int = 64,
        bands: int = 16,
        roles: tuple[Role, ...] | None = None,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if num_perm % bands:
            raise ValueError("num_perm must be a multiple of bands")
        self.threshold = threshold
        self.shingle = shingle
        self.num_perm = num_perm
        self.bands = bands
        self.roles = roles

    def apply(self, examples: Iterable[Example]) -> Iterator[Example]:
        buckets: dict[tuple[int, bytes], list[int]] = {}
        signatures: list[tuple[int, ...]] = []
        rows = self.num_perm // self.bands

        for example in examples:
            signature = self._signature(example)
            if signature is None:
                yield example
                continue
            candidates: set[int] = set()
            keys = []
            for band in range(self.bands):
                chunk = signature[band * rows : (band + 1) * rows]
                key = (band, b"|".join(str(v).encode() for v in chunk))
                keys.append(key)
                candidates.update(buckets.get(key, ()))
            if any(
                self._similarity(signature, signatures[i]) >= self.threshold for i in candidates
            ):
                continue
            position = len(signatures)
            signatures.append(signature)
            for key in keys:
                buckets.setdefault(key, []).append(position)
            yield example

    def _signature(self, example: Example) -> tuple[int, ...] | None:
        text = " ".join(
            m.content for m in example.messages if self.roles is None or m.role in self.roles
        )
        words = _WORD.findall(unicodedata.normalize("NFC", text).casefold())
        if len(words) < self.shingle:
            return None
        shingles = {
            " ".join(words[i : i + self.shingle]) for i in range(len(words) - self.shingle + 1)
        }
        signature = []
        for perm in range(self.num_perm):
            salt = perm.to_bytes(2, "big")
            signature.append(
                min(
                    int.from_bytes(
                        hashlib.blake2b(s.encode("utf-8"), digest_size=8, salt=salt).digest(),
                        "big",
                    )
                    for s in shingles
                )
            )
        return tuple(signature)

    @staticmethod
    def _similarity(left: tuple[int, ...], right: tuple[int, ...]) -> float:
        matches = sum(1 for a, b in zip(left, right, strict=True) if a == b)
        return matches / len(left)


DEFAULT_STEPS: tuple[CurationStep, ...] = (Normalize(), Dedup())


def run_pipeline(
    examples: Iterable[Example], steps: Iterable[CurationStep]
) -> tuple[Iterator[Example], CurationReport]:
    """Compose steps into one lazy stream and return it with a report filled as it drains."""
    report = CurationReport()
    stream: Iterator[Example] = iter(examples)
    for step in steps:
        count = StepCount(step=step.name)
        report.steps.append(count)
        stream = step.apply(_counted(stream, count))
        stream = _counted_out(stream, count)
    return stream, report


def _counted(stream: Iterator[Example], count: StepCount) -> Iterator[Example]:
    for example in stream:
        count.received += 1
        yield example


def _counted_out(stream: Iterator[Example], count: StepCount) -> Iterator[Example]:
    for example in stream:
        count.kept += 1
        yield example
