"""Reading sources from disk: text documents and existing JSONL datasets."""

from __future__ import annotations

import bz2
import glob
import gzip
import hashlib
import json
import logging
import lzma
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

from .domain import Document, Example
from .errors import FormatError, ValidationError
from .formats import DatasetFormat, detect_format, get_format

logger = logging.getLogger("wyra")

# Public datasets ship compressed far more often than not, and a codec traceback is a poor
# way to find out. Handled with the standard library, so the core stays dependency-free.
DECOMPRESSORS: dict[str, Callable[[bytes], bytes]] = {
    ".gz": gzip.decompress,
    ".gzip": gzip.decompress,
    ".xz": lzma.decompress,
    ".lzma": lzma.decompress,
    ".bz2": bz2.decompress,
}
TEXT_OPENERS: dict[str, Any] = {
    ".gz": gzip.open,
    ".gzip": gzip.open,
    ".xz": lzma.open,
    ".lzma": lzma.open,
    ".bz2": bz2.open,
}


def open_text(path: str | Path, *, encoding: str = "utf-8-sig") -> Any:
    """Open a text file for reading, transparently decompressing by extension."""
    target = Path(path)
    opener = TEXT_OPENERS.get(target.suffix.lower())
    if opener is None:
        return target.open(encoding=encoding)
    return opener(target, mode="rt", encoding=encoding)


def decode_bytes(raw: bytes, path: str | Path, *, encoding: str = "utf-8-sig") -> str:
    """Decode file contents, decompressing first when the extension says to."""
    decompress = DECOMPRESSORS.get(Path(path).suffix.lower())
    if decompress is not None:
        try:
            raw = decompress(raw)
        except (OSError, EOFError, lzma.LZMAError) as exc:
            raise FormatError(f"{path} is not a valid archive: {exc}") from exc
    try:
        return raw.decode(encoding)
    except UnicodeDecodeError as exc:
        raise FormatError(
            f"{path} is not {encoding} text. If it is compressed, give it a .gz, .xz or "
            ".bz2 extension so it can be decompressed."
        ) from exc


Source = "str | Path | Document"


def read_document(path: str | Path, *, encoding: str = "utf-8-sig") -> Document:
    """Read one source file. The hash is of the bytes on disk, compressed or not."""
    data = Path(path).read_bytes()
    return Document(
        text=decode_bytes(data, path, encoding=encoding),
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
    """Yield ``(1-based line number, line)`` for every non-blank line, streaming.

    Compressed files are decompressed as they are read, so a multi-gigabyte ``.jsonl.gz``
    never lands in memory.
    """
    try:
        with open_text(path) as handle:
            for line_no, raw in enumerate(handle, start=1):
                line = raw.strip()
                if line:
                    yield line_no, line
    except UnicodeDecodeError as exc:
        raise FormatError(
            f"{path} is not UTF-8 text. If it is compressed, give it a .gz, .xz or .bz2 "
            "extension so it can be decompressed."
        ) from exc
    except (OSError, EOFError, lzma.LZMAError) as exc:
        if isinstance(exc, FileNotFoundError):
            raise
        raise FormatError(f"{path} could not be read: {exc}") from exc


def read_examples(
    path: str | Path,
    *,
    input_format: str | DatasetFormat = "auto",
    on_invalid: str = "raise",
) -> Iterator[Example]:
    """Decode an existing JSONL dataset.

    ``on_invalid="raise"`` stops at the first bad record, naming its line.
    ``on_invalid="skip"`` logs it and keeps going, which is what you want when converting
    somebody else's dataset: a file that is 90 per cent good should still give you 90 per
    cent of a dataset. Use ``validate_jsonl`` first to see what would be dropped.
    """
    if on_invalid not in ("raise", "skip"):
        raise ValueError("on_invalid must be 'raise' or 'skip'")
    fmt: DatasetFormat | None = None if input_format == "auto" else get_format(input_format)
    for line_no, raw in read_jsonl_lines(path):
        try:
            record = json.loads(raw)
            fmt_for_record = fmt if fmt is not None else detect_format(record)
            example = fmt_for_record.decode(record)
        except json.JSONDecodeError as exc:
            if on_invalid == "skip":
                logger.warning("%s:%d skipped, invalid JSON: %s", path, line_no, exc.msg)
                continue
            raise ValidationError(
                f"{path}:{line_no}: invalid JSON: {exc.msg}",
                issues=("invalid_json",),
                index=line_no,
            ) from exc
        except ValidationError as exc:
            if on_invalid == "skip":
                logger.warning("%s:%d skipped: %s", path, line_no, exc)
                continue
            raise ValidationError(
                f"{path}:{line_no}: {exc}", issues=exc.issues, index=line_no
            ) from exc
        except FormatError as exc:
            if on_invalid == "skip":
                logger.warning("%s:%d skipped: %s", path, line_no, exc)
                continue
            raise FormatError(f"{path}:{line_no}: {exc}") from exc
        fmt = fmt_for_record
        yield example
