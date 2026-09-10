"""Reading sources from disk: text documents and existing JSONL datasets."""

from __future__ import annotations

import glob
import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from .domain import Document, Example
from .errors import FormatError, ValidationError
from .formats import DatasetFormat, detect_format, get_format

Source = "str | Path | Document"


def read_document(path: str | Path, *, encoding: str = "utf-8-sig") -> Document:
    data = Path(path).read_bytes()
    return Document(
        text=data.decode(encoding),
        source=str(path),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def read_documents(
    sources: str | Path | Document | Iterable[str | Path | Document],
    *,
    encoding: str = "utf-8-sig",
) -> list[Document]:
    """Expand paths, globs and directories into Documents, in a deterministic order.

    Documents passed in are kept as they are. Files are read as bytes so the manifest can
    record their sha256.
    """
    items: Iterable[str | Path | Document] = (
        [sources] if isinstance(sources, (str, Path, Document)) else sources
    )

    documents: list[Document] = []
    seen: set[Path] = set()
    for item in items:
        if isinstance(item, Document):
            documents.append(item)
            continue
        for path in _expand(item):
            if path in seen:
                continue
            seen.add(path)
            documents.append(read_document(path, encoding=encoding))
    return documents


def _expand(item: str | Path) -> list[Path]:
    pattern = str(item)
    if glob.has_magic(pattern):
        matches = sorted(Path(p) for p in glob.glob(pattern, recursive=True) if Path(p).is_file())
        if not matches:
            raise FileNotFoundError(f"no files match {pattern!r}")
        return matches
    path = Path(pattern)
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.is_file())
    if path.is_file():
        return [path]
    raise FileNotFoundError(f"source not found: {pattern}")


def read_jsonl_lines(path: str | Path) -> Iterator[tuple[int, str]]:
    """Yield ``(1-based line number, line)`` for every non-blank line."""
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if line:
                yield line_no, line


def read_examples(
    path: str | Path, *, input_format: str | DatasetFormat = "auto"
) -> Iterator[Example]:
    """Decode an existing JSONL dataset. Raises ``ValidationError`` with the line number."""
    fmt: DatasetFormat | None = None if input_format == "auto" else get_format(input_format)
    for line_no, raw in read_jsonl_lines(path):
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"{path}:{line_no}: invalid JSON: {exc.msg}",
                issues=("invalid_json",),
                index=line_no,
            ) from exc
        try:
            fmt_for_record = fmt if fmt is not None else detect_format(record)
            example = fmt_for_record.decode(record)
        except ValidationError as exc:
            raise ValidationError(
                f"{path}:{line_no}: {exc}", issues=exc.issues, index=line_no
            ) from exc
        except FormatError as exc:
            raise FormatError(f"{path}:{line_no}: {exc}") from exc
        fmt = fmt_for_record
        yield example
