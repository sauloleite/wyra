from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from wyra.builder import DatasetBuilder, split_examples
from wyra.chunking import ParagraphChunker
from wyra.curation import Dedup, MinLength, Normalize
from wyra.domain import Document, Example
from wyra.errors import GenerationError, ProviderError
from wyra.writers import InMemoryWriter


class EchoGenerator:
    """One example per document, plus optional failures, to exercise the pipeline."""

    name = "echo"

    def __init__(
        self,
        *,
        fail_on: str | None = None,
        error: type[Exception] = GenerationError,
        copies: int = 1,
    ) -> None:
        self.fail_on = fail_on
        self.error = error
        self.copies = copies

    def describe(self) -> Mapping[str, Any]:
        return {"copies": self.copies}

    def generate(self, doc: Document) -> Iterator[Example]:
        if self.fail_on and self.fail_on in doc.text:
            raise self.error("boom")
        for _ in range(self.copies):
            yield Example.qa(f"Explique {doc.ref}?", doc.text, source=doc.ref)


def docs(*texts: str) -> list[Document]:
    return [Document(text, source=f"doc{i}.md", sha256=f"sha{i}") for i, text in enumerate(texts)]


def test_split_is_deterministic_and_seed_dependent() -> None:
    examples = [Example.qa(f"q{i}?", f"a{i}") for i in range(10)]
    train, valid = split_examples(examples, 0.3, seed=42)
    assert (len(train), len(valid)) == (7, 3)
    assert split_examples(examples, 0.3, seed=42) == (train, valid)
    assert split_examples(examples, 0.3, seed=7) != (train, valid)
    assert set(train) | set(valid) == set(examples)
    assert not set(train) & set(valid)


def test_split_edge_cases() -> None:
    examples = [Example.qa("q?", "a")]
    assert split_examples(examples, 0.0) == (examples, [])
    assert split_examples(examples, 0.5) == (examples, [])
    assert split_examples([], 0.5) == ([], [])
    two = [Example.qa("q1?", "a"), Example.qa("q2?", "b")]
    train, valid = split_examples(two, 0.1)
    assert len(valid) == 1  # a non-zero fraction always holds out at least one example
    with pytest.raises(ValueError):
        split_examples(two, 1.0)


def test_build_writes_files_and_manifest(tmp_path: Path) -> None:
    builder = DatasetBuilder(
        EchoGenerator(),
        steps=(Normalize(), Dedup(), MinLength(min_chars=5)),
        validation_fraction=0.5,
        write_lineage=True,
    )
    result = builder.build(docs("primeiro texto", "segundo texto", "hi"), tmp_path)

    assert result.train_path is not None and result.validation_path is not None
    assert result.manifest_path is not None and result.manifest_path.name == "manifest.json"
    assert len(result) == 2
    assert result.counts == {
        "generated": 3,
        "kept": 2,
        "train": 1,
        "validation": 1,
        "errors": 0,
    }
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["generator"] == {"name": "echo", "params": {"copies": 1}}
    assert manifest["chunker"]["name"] == "none"
    assert [s["source"] for s in manifest["sources"]] == ["doc0.md", "doc1.md", "doc2.md"]
    assert manifest["curation"][-1] == {"step": "min_length", "in": 3, "out": 2}
    assert manifest["output"]["format"] == "openai-chat"
    assert {f["name"] for f in manifest["output"]["files"]} == {"train.jsonl", "validation.jsonl"}
    assert all(len(f["sha256"]) == 64 for f in manifest["output"]["files"])
    assert manifest["stats"]["token_counter"] == "approx"
    assert manifest["stats"]["tokens"]["max"] >= manifest["stats"]["tokens"]["min"] > 0

    lineage = [
        json.loads(line)
        for line in (tmp_path / "lineage.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["split"] for row in lineage} == {"train", "validation"}
    assert all(row["source"].startswith("doc") for row in lineage)


def test_build_without_out_dir_writes_nothing(tmp_path: Path) -> None:
    result = DatasetBuilder(EchoGenerator()).build(docs("texto suficiente"), None)
    assert result.train_path is None and result.manifest_path is None
    assert list(tmp_path.iterdir()) == []
    assert len(result.train) == 1


def test_in_memory_writer_and_chunker_are_injectable(tmp_path: Path) -> None:
    writer = InMemoryWriter("sharegpt")
    builder = DatasetBuilder(
        EchoGenerator(),
        chunker=ParagraphChunker(max_chars=20, min_chars=1),
        writer=writer,
    )
    result = builder.build(docs("\n\n".join(["bloco um", "bloco dois", "bloco três"])), tmp_path)
    assert result.manifest.chunker["name"] == "paragraphs"
    assert result.manifest.output["format"] == "sharegpt"
    train = writer.records[str(tmp_path / "train.jsonl")]
    assert train[0]["conversations"][0]["from"] == "human"
    # two chunks: "bloco um" and "bloco dois" fill the budget, "bloco três" starts the next
    assert result.counts["generated"] == 2
    # no JSONL on disk: the writer is in memory, but the manifest is still written
    assert not (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "manifest.json").exists()
    assert "sha256" not in result.manifest.output["files"][0]


def test_generation_errors_are_skipped_and_recorded() -> None:
    builder = DatasetBuilder(EchoGenerator(fail_on="ruim"))
    result = builder.build(docs("texto bom", "texto ruim"), None)
    assert result.counts["errors"] == 1
    assert result.manifest.errors[0]["source"] == "doc1.md#0"
    assert "GenerationError: boom" in result.manifest.errors[0]["error"]
    assert result.counts["kept"] == 1


def test_on_error_raise_and_provider_errors_always_propagate() -> None:
    with pytest.raises(GenerationError):
        DatasetBuilder(EchoGenerator(fail_on="ruim"), on_error="raise").build(docs("ruim"), None)
    with pytest.raises(ProviderError):
        DatasetBuilder(EchoGenerator(fail_on="ruim", error=ProviderError)).build(docs("ruim"), None)
    with pytest.raises(ValueError):
        DatasetBuilder(EchoGenerator(), on_error="ignore")


def test_examples_stream_is_lazy() -> None:
    builder = DatasetBuilder(EchoGenerator(copies=2))
    stream, report = builder.examples(docs("um texto", "outro texto"))
    assert report.generated == 0
    assert isinstance(stream, Iterable)
    first = next(iter(stream))
    assert first.source == "doc0.md#0"
    assert report.generated >= 1


def test_errors_reset_between_builds() -> None:
    builder = DatasetBuilder(EchoGenerator(fail_on="ruim"))
    builder.build(docs("texto ruim"), None)
    result = builder.build(docs("texto bom"), None)
    assert result.manifest.errors == []
