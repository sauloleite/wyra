"""Dataset lineage, produced as a by-product of the build.

What produced this dataset, from which sources, with which parameters, and what was
dropped along the way. Written next to the JSONL so a result can be reproduced and
audited without asking anyone.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_of_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(slots=True)
class Manifest:
    wyra_version: str
    created_at: str = field(default_factory=utc_now)
    generator: dict[str, Any] = field(default_factory=dict)
    chunker: dict[str, Any] = field(default_factory=dict)
    sources: list[dict[str, Any]] = field(default_factory=list)
    curation: list[dict[str, Any]] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)
    split: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "wyra_version": self.wyra_version,
            "created_at": self.created_at,
            "generator": self.generator,
            "chunker": self.chunker,
            "sources": self.sources,
            "curation": self.curation,
            "counts": self.counts,
            "errors": self.errors,
            "split": self.split,
            "output": self.output,
            "stats": self.stats,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, indent=indent, sort_keys=False)

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json() + "\n", encoding="utf-8")
        return destination

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Manifest:
        known = set(cls.__slots__)
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def read(cls, path: str | Path) -> Manifest:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
