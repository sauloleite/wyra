"""The application service: ingest, create, curate, split, save.

``DatasetBuilder`` knows the ports only. Which generator, chunker, curation steps and
writer it runs with is decided by the caller (or by the facade in ``wyra.pipeline``), so
swapping a model provider or an output format never touches this file.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__
from .chunking import NoChunker
from .curation import DEFAULT_STEPS, CurationReport, run_pipeline
from .domain import Document, Example
from .errors import GenerationError, WyraError
from .manifest import Manifest, sha256_of_file
from .ports import Chunker, CurationStep, DatasetWriter, ExampleGenerator, TokenCounter
from .tokens import ApproxTokenCounter, count_example_tokens
from .writers import JsonlWriter

logger = logging.getLogger("wyra")

TRAIN_FILE = "train.jsonl"
VALIDATION_FILE = "validation.jsonl"
LINEAGE_FILE = "lineage.jsonl"
MANIFEST_FILE = "manifest.json"


def split_examples(
    examples: Sequence[Example], fraction: float, seed: int = 42
) -> tuple[list[Example], list[Example]]:
    """Seeded shuffle split. Same input and seed always give the same partition."""
    if not 0.0 <= fraction < 1.0:
        raise ValueError("validation fraction must be in [0, 1)")
    if not fraction or len(examples) < 2:
        return list(examples), []
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    size = max(1, int(len(examples) * fraction))
    validation = {i for i in indices[:size]}
    train = [example for i, example in enumerate(examples) if i not in validation]
    holdout = [example for i, example in enumerate(examples) if i in validation]
    return train, holdout


@dataclass(slots=True)
class BuildResult:
    manifest: Manifest
    report: CurationReport
    train: list[Example]
    validation: list[Example]
    train_path: Path | None = None
    validation_path: Path | None = None
    manifest_path: Path | None = None

    @property
    def counts(self) -> dict[str, int]:
        return self.manifest.counts

    def __len__(self) -> int:
        return len(self.train) + len(self.validation)


class DatasetBuilder:
    """Runs one generator over documents and writes a versioned dataset."""

    def __init__(
        self,
        generator: ExampleGenerator,
        *,
        chunker: Chunker | None = None,
        steps: Sequence[CurationStep] = DEFAULT_STEPS,
        writer: DatasetWriter | None = None,
        validation_fraction: float = 0.0,
        seed: int = 42,
        on_error: str = "skip",
        token_counter: TokenCounter | None = None,
        write_lineage: bool = False,
    ) -> None:
        if on_error not in ("skip", "raise"):
            raise ValueError("on_error must be 'skip' or 'raise'")
        self.generator = generator
        self.chunker = chunker or NoChunker()
        self.steps = tuple(steps)
        self.writer = writer or JsonlWriter()
        self.validation_fraction = validation_fraction
        self.seed = seed
        self.on_error = on_error
        self.token_counter = token_counter or ApproxTokenCounter()
        self.write_lineage = write_lineage
        self._errors: list[dict[str, str]] = []

    def create(self, documents: Iterable[Document]) -> Iterator[Example]:
        """Chunk and generate, lazily. Generation failures follow the ``on_error`` policy."""
        self._errors = []
        for document in documents:
            for chunk in self.chunker.chunk(document):
                try:
                    yield from self.generator.generate(chunk)
                except GenerationError as exc:
                    if self.on_error == "raise":
                        raise
                    logger.warning("skipping %s: %s", chunk.ref, exc)
                    self._errors.append(
                        {"source": chunk.ref, "error": f"{type(exc).__name__}: {exc}"}
                    )

    def examples(self, documents: Iterable[Document]) -> tuple[Iterator[Example], CurationReport]:
        """The lazy create + curate stream, for callers that do not want files."""
        return run_pipeline(self.create(documents), self.steps)

    def build(
        self, documents: Iterable[Document], out_dir: str | Path | None = None
    ) -> BuildResult:
        """Run the whole pipeline. With ``out_dir=None`` nothing is written to disk."""
        documents = list(documents)
        stream, report = self.examples(documents)
        kept = list(stream)  # the split needs the total; datasets for fine-tuning are small
        train, validation = split_examples(kept, self.validation_fraction, self.seed)

        manifest = Manifest(
            wyra_version=__version__,
            generator={"name": self.generator.name, "params": dict(self.generator.describe())},
            chunker={"name": self.chunker.name},
            sources=[
                {"source": d.source, "sha256": d.sha256, "chars": len(d.text)} for d in documents
            ],
            curation=report.as_list(),
            counts={
                "generated": report.generated,
                "kept": report.kept,
                "train": len(train),
                "validation": len(validation),
                "errors": len(self._errors),
            },
            errors=list(self._errors),
            split={"validation_fraction": self.validation_fraction, "seed": self.seed},
            output={"format": self.writer.format, "files": []},
            stats=self._token_stats(kept),
        )
        result = BuildResult(manifest=manifest, report=report, train=train, validation=validation)
        if out_dir is not None:
            self._save(result, Path(out_dir))
        return result

    def _save(self, result: BuildResult, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        files: list[dict[str, Any]] = []

        result.train_path = out_dir / TRAIN_FILE
        written = self.writer.write(result.train, result.train_path)
        files.append(self._file_entry(result.train_path, written))

        if result.validation:
            result.validation_path = out_dir / VALIDATION_FILE
            written = self.writer.write(result.validation, result.validation_path)
            files.append(self._file_entry(result.validation_path, written))

        if self.write_lineage:
            self._write_lineage(result, out_dir / LINEAGE_FILE)

        result.manifest.output["files"] = files
        result.manifest_path = result.manifest.write(out_dir / MANIFEST_FILE)
        logger.info("wrote %s example(s) to %s", result.manifest.counts["kept"], out_dir)

    @staticmethod
    def _file_entry(path: Path, records: int) -> dict[str, Any]:
        entry: dict[str, Any] = {"name": path.name, "records": records}
        if path.exists():
            entry["sha256"] = sha256_of_file(path)
        return entry

    @staticmethod
    def _write_lineage(result: BuildResult, path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for split, examples in (("train", result.train), ("validation", result.validation)):
                for index, example in enumerate(examples):
                    handle.write(
                        json.dumps(
                            {"split": split, "index": index, "source": example.source},
                            ensure_ascii=False,
                        )
                    )
                    handle.write("\n")

    def _token_stats(self, examples: Sequence[Example]) -> dict[str, Any]:
        counts = [count_example_tokens(e, self.token_counter) for e in examples]
        return {
            "token_counter": self.token_counter.name,
            "tokens": {
                "min": min(counts) if counts else 0,
                "max": max(counts) if counts else 0,
                "mean": round(sum(counts) / len(counts), 1) if counts else 0.0,
                "total": sum(counts),
            },
        }


__all__ = ["BuildResult", "DatasetBuilder", "WyraError", "split_examples"]
