from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from wyra.domain import Document, Example
from wyra.errors import FormatError, ValidationError
from wyra.readers import read_document, read_documents, read_examples, read_jsonl_lines


def test_read_document_records_sha256_and_strips_bom(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_bytes("﻿Olá".encode())
    doc = read_document(path)
    assert doc.text == "Olá"
    assert doc.source == str(path)
    assert doc.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert doc.index == 0


def test_read_documents_expands_globs_dirs_and_keeps_order(tmp_path: Path) -> None:
    (tmp_path / "b.md").write_text("b", encoding="utf-8")
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.txt").write_text("c", encoding="utf-8")

    docs = read_documents(str(tmp_path / "*.md"))
    assert [d.text for d in docs] == ["a", "b"]

    docs = read_documents(tmp_path)
    assert [d.text for d in docs] == ["a", "b", "c"]

    inline = Document("inline")
    docs = read_documents([inline, tmp_path / "a.md", tmp_path / "a.md"])
    assert docs == [inline, docs[1]]
    assert docs[1].text == "a"

    with pytest.raises(FileNotFoundError):
        read_documents(tmp_path / "missing.md")
    with pytest.raises(FileNotFoundError):
        read_documents(str(tmp_path / "*.nope"))


def test_read_jsonl_lines_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    path.write_text('{"a": 1}\n\n  \n{"b": 2}\n', encoding="utf-8")
    assert list(read_jsonl_lines(path)) == [(1, '{"a": 1}'), (4, '{"b": 2}')]


def test_read_examples_auto_detects_and_reports_line(fixtures: Path, tmp_path: Path) -> None:
    examples = list(read_examples(fixtures / "sharegpt.jsonl"))
    assert len(examples) == 2 and examples[0].system == "Seja breve."
    alpaca = list(read_examples(fixtures / "alpaca.jsonl", input_format="alpaca"))
    assert alpaca[0] == Example.qa("Traduza para o inglês.\n\ncódigo limpo", "clean code")

    with pytest.raises(ValidationError) as info:
        list(read_examples(fixtures / "bad.jsonl"))
    assert info.value.index == 1 and "data_type" in info.value.issues

    broken = tmp_path / "broken.jsonl"
    broken.write_text(
        '{"messages": [{"role": "user", "content": "q"}, '
        '{"role": "assistant", "content": "a"}]}\n{oops\n',
        encoding="utf-8",
    )
    with pytest.raises(ValidationError) as info:
        list(read_examples(broken))
    assert info.value.index == 2 and info.value.issues == ("invalid_json",)


def compress(path: Path, data: bytes, suffix: str) -> Path:
    import bz2
    import gzip
    import lzma

    packers = {".gz": gzip.compress, ".xz": lzma.compress, ".bz2": bz2.compress}
    target = path.with_name(path.name + suffix)
    target.write_bytes(packers[suffix](data))
    return target


@pytest.mark.parametrize("suffix", [".gz", ".xz", ".bz2"])
def test_compressed_sources_are_read_transparently(tmp_path: Path, suffix: str) -> None:
    import hashlib

    payload = "# Título\n\nCorpo com acento: ação.\n".encode()
    packed = compress(tmp_path / "doc.md", payload, suffix)

    doc = read_document(packed)
    assert doc.text == payload.decode()
    # the hash is of the bytes on disk, so lineage matches the artefact you actually have
    assert doc.sha256 == hashlib.sha256(packed.read_bytes()).hexdigest()
    assert read_documents(packed)[0].text == payload.decode()


@pytest.mark.parametrize("suffix", [".gz", ".xz", ".bz2"])
def test_compressed_jsonl_streams_and_validates(
    fixtures: Path, tmp_path: Path, suffix: str
) -> None:
    from wyra.validation import validate_jsonl

    packed = compress(tmp_path / "good.jsonl", (fixtures / "good.jsonl").read_bytes(), suffix)

    assert len(list(read_jsonl_lines(packed))) == 3
    assert len(list(read_examples(packed))) == 3
    report = validate_jsonl(packed)
    assert report.ok and report.total == 3


def test_a_binary_file_says_what_to_do_instead_of_a_codec_traceback(tmp_path: Path) -> None:
    binary = tmp_path / "weights.bin"
    binary.write_bytes(b"\x00\x8b\xff" * 50)
    with pytest.raises(FormatError, match=r"\.gz, \.xz or \.bz2"):
        list(read_jsonl_lines(binary))
    with pytest.raises(FormatError, match=r"\.gz, \.xz or \.bz2"):
        read_document(binary)


def test_a_corrupt_archive_is_reported_as_such(tmp_path: Path) -> None:
    broken = tmp_path / "data.jsonl.gz"
    broken.write_bytes(b"not actually gzip")
    with pytest.raises(FormatError, match="not a valid archive"):
        read_document(broken)
    with pytest.raises(FormatError):
        list(read_jsonl_lines(broken))


def test_skip_invalid_keeps_the_records_that_are_fine(fixtures: Path) -> None:
    kept = list(read_examples(fixtures / "bad.jsonl", on_invalid="skip"))
    assert len(kept) == 1
    assert kept[0].messages[0].content == "valid"

    with pytest.raises(ValidationError):
        list(read_examples(fixtures / "bad.jsonl"))
    with pytest.raises(ValueError, match="on_invalid"):
        list(read_examples(fixtures / "good.jsonl", on_invalid="ignore"))


def test_skip_invalid_logs_every_dropped_line(fixtures: Path, caplog) -> None:
    import logging

    with caplog.at_level(logging.WARNING, logger="wyra"):
        list(read_examples(fixtures / "bad.jsonl", on_invalid="skip"))
    dropped = [record.getMessage() for record in caplog.records]
    assert len(dropped) == 10
    assert any("invalid JSON" in message for message in dropped)
