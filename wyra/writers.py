"""Dataset writers: JSONL on disk and an in-memory fake for tests."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .domain import Example
from .formats import DatasetFormat, get_format


class JsonlWriter:
    """One JSON object per line, UTF-8, non-ASCII kept as is."""

    def __init__(self, fmt: str | DatasetFormat = "openai-chat") -> None:
        self._format = get_format(fmt)
        self.format = self._format.name

    def write(self, examples: Iterable[Example], destination: str | Path) -> int:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for example in examples:
                handle.write(json.dumps(self._format.encode(example), ensure_ascii=False))
                handle.write("\n")
                count += 1
        return count


class InMemoryWriter:
    """Fake writer: keeps the encoded records per destination for assertions."""

    def __init__(self, fmt: str | DatasetFormat = "openai-chat") -> None:
        self._format = get_format(fmt)
        self.format = self._format.name
        self.records: dict[str, list[dict[str, Any]]] = {}

    def write(self, examples: Iterable[Example], destination: str | Path) -> int:
        records = [self._format.encode(example) for example in examples]
        self.records[str(destination)] = records
        return len(records)
