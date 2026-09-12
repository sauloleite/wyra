"""The facade: three functions that cover the common cases in a few lines each.

Everything here is assembly. The behaviour lives in the generators, the curation steps
and the builder, so nothing below decides anything a caller cannot override.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .builder import BuildResult, DatasetBuilder
from .chunking import NoChunker, ParagraphChunker
from .curation import Dedup, Normalize, TokenBudget
from .domain import Document, Example
from .errors import ConfigError
from .generators import create as create_generator
from .generators import for_source, needs_provider
from .ports import Chunker, CompletionProvider, CurationStep, ExampleGenerator, TokenCounter
from .readers import read_documents, read_examples
from .tokens import get_counter
from .validation import validate_jsonl
from .writers import JsonlWriter

__all__ = ["build_dataset", "convert_jsonl", "validate_jsonl"]

# Measured, not guessed: qwen2.5-0.5b answers cleanly up to about 650 characters of source
# (280 prompt tokens) and degenerates into whitespace at 910. Smaller chunks also raise the
# signal-to-noise ratio of the context, which helps capable models too.
DEFAULT_CHUNK_CHARS = 600


def build_dataset(
    sources: str | Path | Document | Iterable[str | Path | Document],
    out_dir: str | Path | None = None,
    *,
    generator: str | ExampleGenerator = "auto",
    provider: str | CompletionProvider | None = None,
    model: str | None = None,
    lang: str = "en",
    output_format: str = "openai-chat",
    system_prompt: str | None = None,
    validation_fraction: float = 0.0,
    seed: int = 42,
    chunker: Chunker | None = None,
    steps: Sequence[CurationStep] | None = None,
    max_tokens: int | None = None,
    dedupe: bool = True,
    token_counter: str | TokenCounter | None = None,
    on_error: str = "skip",
    write_lineage: bool = False,
    **generator_kwargs: Any,
) -> BuildResult:
    """Read sources, generate examples, curate them and write a versioned dataset.

    ``generator="auto"`` picks a deterministic converter from the file extension and
    never picks one that would call a model.
    """
    documents = read_documents(sources)
    resolved = _resolve_generator(
        generator, documents, provider, model, lang, system_prompt, generator_kwargs
    )
    counter = get_counter(token_counter)
    builder = DatasetBuilder(
        resolved,
        chunker=chunker or _default_chunker(resolved),
        steps=steps if steps is not None else _default_steps(dedupe, max_tokens, counter),
        writer=JsonlWriter(output_format),
        validation_fraction=validation_fraction,
        seed=seed,
        on_error=on_error,
        token_counter=counter,
        write_lineage=write_lineage,
    )
    return builder.build(documents, out_dir)


def convert_jsonl(
    source: str | Path,
    destination: str | Path,
    *,
    input_format: str = "auto",
    output_format: str = "openai-chat",
    dedupe: bool = True,
    validation_fraction: float = 0.0,
    seed: int = 42,
    on_invalid: str = "raise",
    token_counter: str | TokenCounter | None = None,
) -> BuildResult:
    """Re-encode an existing dataset into another format, normalizing and deduplicating."""
    examples = list(read_examples(source, input_format=input_format, on_invalid=on_invalid))
    return build_examples(
        examples,
        destination,
        output_format=output_format,
        dedupe=dedupe,
        validation_fraction=validation_fraction,
        seed=seed,
        source=str(source),
        token_counter=token_counter,
    )


def build_examples(
    examples: Iterable[Example],
    destination: str | Path,
    *,
    output_format: str = "openai-chat",
    dedupe: bool = True,
    validation_fraction: float = 0.0,
    seed: int = 42,
    source: str = "<examples>",
    token_counter: str | TokenCounter | None = None,
) -> BuildResult:
    """Curate and write examples you already have. The destination is a file, not a folder."""
    path = Path(destination)
    writer = JsonlWriter(output_format)
    counter = get_counter(token_counter)
    builder = DatasetBuilder(
        _PreBuilt(list(examples), source),
        steps=_default_steps(dedupe, None, counter),
        writer=writer,
        validation_fraction=validation_fraction,
        seed=seed,
        token_counter=counter,
    )
    # the carrier document is never read by _PreBuilt, but it must not be empty:
    # NoChunker drops blank documents, which is the right behaviour for real sources.
    result = builder.build([Document(source, source=source)], None)

    path.parent.mkdir(parents=True, exist_ok=True)
    result.train_path = path
    files = [{"name": path.name, "records": writer.write(result.train, path)}]
    if result.validation:
        result.validation_path = path.with_name(f"{path.stem}.validation{path.suffix}")
        files.append(
            {
                "name": result.validation_path.name,
                "records": writer.write(result.validation, result.validation_path),
            }
        )
    result.manifest.output["files"] = files
    result.manifest_path = result.manifest.write(path.with_suffix(path.suffix + ".manifest.json"))
    return result


class _PreBuilt:
    """Adapter that lets already-built examples flow through the same pipeline."""

    name = "prebuilt"

    def __init__(self, examples: Sequence[Example], source: str) -> None:
        self._examples = examples
        self._source = source

    def describe(self) -> dict[str, Any]:
        return {"source": self._source, "count": len(self._examples)}

    def generate(self, doc: Document) -> Iterable[Example]:
        return self._examples


def _resolve_generator(
    generator: str | ExampleGenerator,
    documents: Sequence[Document],
    provider: str | CompletionProvider | None,
    model: str | None,
    lang: str,
    system_prompt: str | None,
    kwargs: dict[str, Any],
) -> ExampleGenerator:
    if not isinstance(generator, str):
        return generator
    name = generator
    if name == "auto":
        if not documents:
            raise ConfigError("no sources to read")
        name = for_source(documents[0].source)
    if system_prompt is not None:
        kwargs.setdefault("system", system_prompt)
    if needs_provider(name):
        kwargs["provider"] = _resolve_provider(provider, model)
    return create_generator(name, lang=lang, **kwargs)


def _resolve_provider(
    provider: str | CompletionProvider | None, model: str | None
) -> CompletionProvider:
    from . import providers  # imported here so the core never pulls in an SDK

    if provider is None or isinstance(provider, str):
        return providers.create(provider, model=model)
    return provider


def _default_chunker(generator: ExampleGenerator) -> Chunker:
    """Only model-backed generation needs chunks; converters want the whole document."""
    return (
        ParagraphChunker(max_chars=DEFAULT_CHUNK_CHARS)
        if needs_provider(generator.name)
        else NoChunker()
    )


def _default_steps(
    dedupe: bool, max_tokens: int | None, counter: TokenCounter
) -> tuple[CurationStep, ...]:
    steps: list[CurationStep] = [Normalize()]
    if dedupe:
        steps.append(Dedup())
    if max_tokens:
        steps.append(TokenBudget(max_tokens, counter=counter))
    return tuple(steps)
