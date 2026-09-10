from __future__ import annotations

from pathlib import Path

from wyra.domain import Document
from wyra.generators import MarkdownSectionGenerator
from wyra.prompts import PT_BR


def read(fixtures: Path) -> Document:
    return Document((fixtures / "clean_code.md").read_text(encoding="utf-8"), source="cc.md")


def test_sections_become_questions_with_parent_context(fixtures: Path) -> None:
    examples = list(MarkdownSectionGenerator(prompts=PT_BR).generate(read(fixtures)))
    questions = [e.messages[0].content for e in examples]
    assert "Explique: Código limpo." in questions
    assert "No contexto de Código limpo, explique: Nomes são muito importantes." in questions
    assert "No contexto de Código limpo > Funções pequenas, explique: Por que isso acontece?." in (
        questions
    )
    assert all(e.source == "cc.md#0" for e in examples)


def test_short_sections_are_skipped_and_bodies_are_complete(fixtures: Path) -> None:
    examples = list(MarkdownSectionGenerator(prompts=PT_BR).generate(read(fixtures)))
    titles = [e.messages[0].content for e in examples]
    assert not any("Curta" in title for title in titles)
    body = next(
        e.messages[-1].content for e in examples if "Regra de Escoteiro" in e.messages[0].content
    )
    assert body.startswith("Deixe o código mais limpo") and body.endswith("já contam.")


def test_hash_inside_a_code_fence_is_not_a_heading(fixtures: Path) -> None:
    examples = list(
        MarkdownSectionGenerator(prompts=PT_BR, min_body_chars=1).generate(read(fixtures))
    )
    titles = [e.messages[0].content for e in examples]
    assert not any("cerca de código" in title for title in titles)
    functions = next(e for e in examples if "Funções pequenas" in e.messages[0].content)
    assert "def soma(a, b):" in functions.messages[-1].content


def test_options_system_levels_and_parents() -> None:
    doc = Document("# Um\n\ncorpo do um\n\n## Dois\n\ncorpo do dois", source="d.md")
    plain = list(MarkdownSectionGenerator(min_body_chars=1, include_parents=False).generate(doc))
    assert [e.messages[0].content for e in plain] == ["Explain: Um.", "Explain: Dois."]

    with_system = list(
        MarkdownSectionGenerator(min_body_chars=1, system="Seja breve.").generate(doc)
    )
    assert with_system[0].system == "Seja breve."

    top_only = list(MarkdownSectionGenerator(min_body_chars=1, max_level=1).generate(doc))
    assert len(top_only) == 1


def test_no_headings_and_empty_text_yield_nothing() -> None:
    assert list(MarkdownSectionGenerator().generate(Document("just prose, no heading"))) == []
    assert list(MarkdownSectionGenerator().generate(Document(""))) == []


def test_describe_is_manifest_ready() -> None:
    generator = MarkdownSectionGenerator(prompts=PT_BR, min_body_chars=10)
    assert generator.name == "markdown-sections"
    assert dict(generator.describe()) == {
        "lang": "pt-br",
        "min_body_chars": 10,
        "include_parents": True,
        "max_level": 6,
        "system": None,
    }


def test_closing_hashes_are_stripped_from_titles() -> None:
    doc = Document("## Título ##\n\ncorpo suficiente para passar", source="d.md")
    example = next(iter(MarkdownSectionGenerator(min_body_chars=1).generate(doc)))
    assert example.messages[0].content == "Explain: Título."
