from __future__ import annotations

import pytest

from wyra.domain import Document, Example
from wyra.errors import ConfigError
from wyra.generators import (
    GENERATORS,
    MarkdownSectionGenerator,
    available,
    create,
    for_source,
    register,
)


def test_create_by_name_passes_the_language_preset() -> None:
    generator = create("markdown", lang="pt-br", min_body_chars=5)
    assert isinstance(generator, MarkdownSectionGenerator)
    assert dict(generator.describe())["lang"] == "pt-br"
    assert dict(generator.describe())["min_body_chars"] == 5
    assert dict(create("cloze", lang="pt-br").describe())["lang"] == "pt-br"


def test_create_does_not_pass_prompts_to_generators_that_reject_them() -> None:
    generator = create("template", lang="pt-br", user="{a}?", assistant="{b}")
    assert dict(generator.describe())["user"] == "{a}?"


def test_unknown_generator_and_language_list_the_options() -> None:
    with pytest.raises(ConfigError, match="unknown generator"):
        create("magic")
    with pytest.raises(ConfigError, match="unknown language"):
        create("markdown", lang="klingon")


def test_for_source_maps_extensions_and_refuses_to_guess() -> None:
    assert for_source("docs/guide.md") == "markdown"
    assert for_source("data/rows.CSV") == "template"
    assert for_source("data/rows.jsonl") == "template"
    with pytest.raises(ConfigError, match="cannot guess"):
        for_source("notes.txt")


def test_register_extends_the_registry() -> None:
    class Custom:
        name = "custom"

        def describe(self) -> dict[str, object]:
            return {}

        def generate(self, doc: Document) -> list[Example]:
            return [Example.qa("q?", doc.text)]

    try:
        register("custom", Custom)
        assert "custom" in available()
        assert create("custom").name == "custom"
    finally:
        GENERATORS.pop("custom", None)
    assert "custom" not in available()
