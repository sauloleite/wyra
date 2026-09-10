"""Tabular records into examples through templates. No model, fully deterministic.

This is the most productive path in the library: you write the phrasing once and every
row of a CSV, JSON or JSONL file becomes an example. It is how FLAN and T0 were built, and
how the training data for instruction-generation models is bootstrapped.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..domain import Document, Example
from ..errors import ConfigError, FormatError

# Column names recognized when no template is given, in priority order.
# Most specific first: "instruction"/"output" must win over the generic "input"/"output".
COLUMN_PAIRS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("instruction", "instrucao", "instrução"),
        ("output", "response", "resposta", "saida", "saída", "completion"),
    ),
    (
        ("question", "pergunta", "prompt"),
        ("answer", "resposta", "output", "completion", "response"),
    ),
    (("input", "entrada"), ("output", "saida", "saída", "target")),
)
CONTEXT_COLUMNS: tuple[str, ...] = ("input", "context", "contexto", "entrada")
SYSTEM_COLUMNS: tuple[str, ...] = ("system", "sistema", "system_prompt")


class TemplateGenerator:
    """Render every record with a user and an assistant template.

    Without templates, common column names are detected (``question``/``answer``,
    ``instruction``/``output`` and their Portuguese equivalents).
    """

    name = "template"

    def __init__(
        self,
        user: str | None = None,
        assistant: str | None = None,
        *,
        system: str | None = None,
        skip_incomplete: bool = False,
    ) -> None:
        if bool(user) != bool(assistant):
            raise ConfigError("provide both 'user' and 'assistant' templates, or neither")
        self.user = user
        self.assistant = assistant
        self.system = system
        self.skip_incomplete = skip_incomplete

    def describe(self) -> Mapping[str, Any]:
        return {
            "user": self.user,
            "assistant": self.assistant,
            "system": self.system,
            "skip_incomplete": self.skip_incomplete,
        }

    def generate(self, doc: Document) -> Iterator[Example]:
        records = list(parse_records(doc))
        if not records:
            return
        user, assistant, context, system_col = self._resolve(records[0])
        for record in records:
            try:
                question = user.format_map(_Safe(record))
                answer = assistant.format_map(_Safe(record))
            except KeyError as exc:
                raise FormatError(f"template refers to a missing column: {exc}") from exc
            if context and str(record.get(context, "")).strip():
                question = f"{question}\n\n{record[context]}"
            if not question.strip() or not answer.strip():
                if self.skip_incomplete:
                    continue
                raise FormatError(f"record produced an empty example: {record!r}")
            system = self.system
            if system_col and str(record.get(system_col, "")).strip():
                system = str(record[system_col])
            yield Example.qa(question.strip(), answer.strip(), system=system, source=doc.ref)

    def _resolve(self, sample: Mapping[str, Any]) -> tuple[str, str, str | None, str | None]:
        system_col = next((c for c in SYSTEM_COLUMNS if c in sample), None)
        if self.user and self.assistant:
            return self.user, self.assistant, None, system_col
        columns = {str(key).strip().lower(): key for key in sample}
        for question_names, answer_names in COLUMN_PAIRS:
            question = next((columns[n] for n in question_names if n in columns), None)
            answer = next((columns[n] for n in answer_names if n in columns), None)
            if question and answer and question != answer:
                context = next(
                    (
                        columns[c]
                        for c in CONTEXT_COLUMNS
                        if c in columns and columns[c] != question
                    ),
                    None,
                )
                return f"{{{question}}}", f"{{{answer}}}", context, system_col
        raise ConfigError(
            "could not detect question/answer columns in "
            f"{sorted(columns)}; pass user= and assistant= templates"
        )


class _Safe(dict):  # type: ignore[type-arg]
    """Mapping for ``format_map`` that keeps a KeyError informative."""

    def __missing__(self, key: str) -> Any:
        raise KeyError(key)


def parse_records(doc: Document) -> Iterator[Mapping[str, Any]]:
    """Read CSV, JSON or JSONL records out of a Document, guided by its source extension."""
    text = doc.text.strip()
    if not text:
        return
    suffix = Path(doc.source).suffix.lower()
    if suffix == ".csv" or (suffix not in (".json", ".jsonl", ".ndjson") and _looks_like_csv(text)):
        yield from _read_csv(text)
    elif suffix in (".jsonl", ".ndjson") or (suffix != ".json" and text.startswith("{")):
        yield from _read_jsonl(text)
    else:
        yield from _read_json(text)


def _looks_like_csv(text: str) -> bool:
    head = text.splitlines()[0]
    return "," in head or ";" in head or "\t" in head


def _read_csv(text: str) -> Iterator[Mapping[str, Any]]:
    try:
        dialect: Any = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    for row in csv.DictReader(io.StringIO(text), dialect=dialect):
        yield {k: ("" if v is None else v) for k, v in row.items() if k is not None}


def _read_jsonl(text: str) -> Iterator[Mapping[str, Any]]:
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FormatError(f"line {number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise FormatError(f"line {number}: expected a JSON object")
        yield record


def _read_json(text: str) -> Iterator[Mapping[str, Any]]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise FormatError(f"invalid JSON: {exc.msg}") from exc
    if isinstance(data, Mapping):
        for key in ("data", "records", "rows", "items"):
            if isinstance(data.get(key), Sequence):
                data = data[key]
                break
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
        raise FormatError("expected a JSON array of objects")
    for record in data:
        if not isinstance(record, Mapping):
            raise FormatError("expected a JSON array of objects")
        yield record
