"""Generator registry: pick one by name, or let the source extension decide.

New generators register themselves without this module knowing about them, and without
``build_dataset`` growing another branch.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ..errors import ConfigError
from ..ports import ExampleGenerator
from ..prompts import Prompts, by_lang
from .llm import InstructionGenerator, LLMExampleGenerator, QAPairGenerator
from .markdown import MarkdownSectionGenerator
from .qa_pairs import QAPairExtractor
from .selfsupervised import ClozeGenerator, ContinuationGenerator
from .template import TemplateGenerator

GeneratorFactory = Callable[..., ExampleGenerator]

GENERATORS: dict[str, GeneratorFactory] = {
    "markdown": MarkdownSectionGenerator,
    "qa": QAPairExtractor,
    "template": TemplateGenerator,
    "continuation": ContinuationGenerator,
    "cloze": ClozeGenerator,
    "llm-qa": QAPairGenerator,
    "llm-instruction": InstructionGenerator,
}

# Generators that need a CompletionProvider as their first argument.
NEEDS_PROVIDER = frozenset({"llm-qa", "llm-instruction"})

# Which generator a file extension implies when none is given.
BY_EXTENSION: dict[str, str] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".csv": "template",
    ".tsv": "template",
    ".json": "template",
    ".jsonl": "template",
    ".ndjson": "template",
}

# Generators that take a ``prompts`` keyword.
_LOCALIZED = frozenset({"markdown", "continuation", "cloze", "llm-qa", "llm-instruction"})


def register(name: str, factory: GeneratorFactory) -> None:
    """Add a generator to the registry, so ``--generator <name>`` finds it."""
    GENERATORS[name] = factory


def available() -> list[str]:
    return sorted(GENERATORS)


def create(
    name: str, *, lang: str = "en", prompts: Prompts | None = None, **kwargs: Any
) -> ExampleGenerator:
    """Build a generator by name, passing the language preset when it accepts one."""
    try:
        factory = GENERATORS[name]
    except KeyError:
        raise ConfigError(
            f"unknown generator {name!r}; available: {', '.join(available())}"
        ) from None
    if name in _LOCALIZED:
        kwargs.setdefault("prompts", prompts or by_lang(lang))
    return factory(**kwargs)


def for_source(source: str | Path) -> str:
    """Guess a generator from a file extension. Never guesses one that calls a model."""
    suffix = Path(source).suffix.lower()
    try:
        return BY_EXTENSION[suffix]
    except KeyError:
        raise ConfigError(
            f"cannot guess a generator for {suffix or 'this source'!r}: choose one of "
            f"{', '.join(available())} (plain prose usually means 'qa', 'continuation', "
            "'cloze', or an LLM-backed generator)"
        ) from None


def describe_all(generators: list[ExampleGenerator]) -> Mapping[str, Any]:
    return {g.name: dict(g.describe()) for g in generators}


__all__ = [
    "BY_EXTENSION",
    "GENERATORS",
    "NEEDS_PROVIDER",
    "ClozeGenerator",
    "ContinuationGenerator",
    "InstructionGenerator",
    "LLMExampleGenerator",
    "MarkdownSectionGenerator",
    "QAPairExtractor",
    "QAPairGenerator",
    "TemplateGenerator",
    "available",
    "create",
    "for_source",
    "needs_provider",
    "register",
]


def needs_provider(name: str) -> bool:
    """Whether this generator will call a model, and therefore needs a provider."""
    return name in NEEDS_PROVIDER
