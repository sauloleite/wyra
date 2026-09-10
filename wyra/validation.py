"""Validation of external datasets: counts every issue instead of stopping at the first."""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .domain import Example
from .errors import FormatError, ValidationError
from .formats import DatasetFormat, detect_format, get_format
from .ports import TokenCounter
from .readers import read_jsonl_lines
from .tokens import ApproxTokenCounter, count_example_tokens

DEFAULT_TOKEN_BUDGET = 16385


@dataclass(slots=True)
class TokenStats:
    counter: str
    count: int = 0
    min: int = 0
    max: int = 0
    mean: float = 0.0
    over_budget: int = 0
    budget: int = DEFAULT_TOKEN_BUDGET

    def as_dict(self) -> dict[str, Any]:
        return {
            "token_counter": self.counter,
            "budget": self.budget,
            "tokens": {
                "min": self.min,
                "max": self.max,
                "mean": round(self.mean, 1),
                "over_budget": self.over_budget,
            },
        }


@dataclass(slots=True)
class ValidationReport:
    format: str
    path: str | None = None
    total: int = 0
    valid: int = 0
    issues: dict[str, int] = field(default_factory=dict)
    invalid: list[tuple[int, tuple[str, ...]]] = field(default_factory=list)
    tokens: TokenStats | None = None
    max_listed: int = 20
    _token_counts: list[int] = field(default_factory=list, repr=False)

    @property
    def ok(self) -> bool:
        return self.total > 0 and self.valid == self.total

    def add_valid(self, example: Example, counter: TokenCounter | None = None) -> None:
        self.total += 1
        self.valid += 1
        if counter is not None:
            self._token_counts.append(count_example_tokens(example, counter))

    def add_invalid(self, index: int, issues: tuple[str, ...]) -> None:
        self.total += 1
        for issue in issues:
            self.issues[issue] = self.issues.get(issue, 0) + 1
        if len(self.invalid) < self.max_listed:
            self.invalid.append((index, issues))

    def finish(self, counter: TokenCounter | None, budget: int) -> ValidationReport:
        if counter is not None:
            counts = self._token_counts
            self.tokens = TokenStats(
                counter=counter.name,
                count=len(counts),
                min=min(counts) if counts else 0,
                max=max(counts) if counts else 0,
                mean=statistics.fmean(counts) if counts else 0.0,
                over_budget=sum(1 for c in counts if c > budget),
                budget=budget,
            )
        return self

    def summary(self) -> str:
        lines = [
            f"{self.path or '<records>'}: {self.valid}/{self.total} valid records ({self.format})"
        ]
        for issue, count in sorted(self.issues.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"  {issue}: {count}")
        if self.invalid:
            shown = ", ".join(str(index) for index, _ in self.invalid)
            lines.append(f"  first invalid lines: {shown}")
        if self.tokens is not None and self.tokens.count:
            t = self.tokens
            lines.append(
                f"  tokens ({t.counter}): min {t.min}, max {t.max}, mean {t.mean:.1f}, "
                f"over budget ({t.budget}): {t.over_budget}"
            )
        return "\n".join(lines)


def validate_records(
    records: Iterable[tuple[int, Any]],
    fmt: str | DatasetFormat | None = None,
    *,
    counter: TokenCounter | None = None,
    budget: int = DEFAULT_TOKEN_BUDGET,
    max_listed: int = 20,
    path: str | None = None,
) -> ValidationReport:
    """Validate ``(index, record)`` pairs. ``fmt=None`` detects the format from the first record."""
    resolved = None if fmt is None else get_format(fmt)
    report = ValidationReport(
        format=resolved.name if resolved else "auto", path=path, max_listed=max_listed
    )
    for index, record in records:
        if isinstance(record, ValidationError):
            report.add_invalid(index, record.issues or ("invalid_record",))
            continue
        try:
            fmt_for_record = resolved if resolved is not None else detect_format(record)
            example = fmt_for_record.decode(record)
        except ValidationError as exc:
            report.add_invalid(index, exc.issues or ("invalid_record",))
            continue
        except FormatError:
            report.add_invalid(index, ("unknown_format",))
            continue
        if resolved is None:
            resolved = fmt_for_record
            report.format = resolved.name
        report.add_valid(example, counter)
    return report.finish(counter, budget)


def parse_jsonl(path: str | Path) -> Iterable[tuple[int, Any]]:
    """Yield ``(line_number, object)``; a bad line yields a ``ValidationError`` instead."""
    for line_no, raw in read_jsonl_lines(path):
        try:
            yield line_no, json.loads(raw)
        except json.JSONDecodeError as exc:
            yield (
                line_no,
                ValidationError(
                    f"line {line_no}: invalid JSON: {exc.msg}",
                    issues=("invalid_json",),
                    index=line_no,
                ),
            )


def validate_jsonl(
    path: str | Path,
    *,
    input_format: str = "auto",
    counter: TokenCounter | None = None,
    budget: int = DEFAULT_TOKEN_BUDGET,
    max_listed: int = 20,
) -> ValidationReport:
    """Validate a JSONL file and return a report with per-issue counts and token stats."""
    fmt = None if input_format == "auto" else input_format
    return validate_records(
        parse_jsonl(path),
        fmt,
        counter=counter if counter is not None else ApproxTokenCounter(),
        budget=budget,
        max_listed=max_listed,
        path=str(path),
    )
