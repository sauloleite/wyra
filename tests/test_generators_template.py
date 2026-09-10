from __future__ import annotations

import json
from pathlib import Path

import pytest

from wyra.domain import Document
from wyra.errors import ConfigError, FormatError
from wyra.generators import TemplateGenerator


def test_explicit_templates_render_every_row() -> None:
    doc = Document("nome,preco\nCafé,10\nChá,5\n", source="p.csv")
    generator = TemplateGenerator(
        user="Qual o preço de {nome}?", assistant="{nome} custa R$ {preco}."
    )
    examples = list(generator.generate(doc))
    assert [e.messages[0].content for e in examples] == [
        "Qual o preço de Café?",
        "Qual o preço de Chá?",
    ]
    assert examples[0].messages[-1].content == "Café custa R$ 10."
    assert examples[0].source == "p.csv#0"


def test_column_detection_for_csv(fixtures: Path) -> None:
    doc = Document((fixtures / "qa.csv").read_text(encoding="utf-8"), source="qa.csv")
    examples = list(TemplateGenerator().generate(doc))
    assert len(examples) == 3
    assert examples[0].messages[0].content == "O que é DRY?"
    assert examples[2].messages[-1].content.startswith("Deixe o código mais limpo")


def test_alpaca_columns_merge_the_input_into_the_question() -> None:
    rows = [{"instruction": "Traduza.", "input": "código limpo", "output": "clean code"}]
    doc = Document(json.dumps(rows), source="a.json")
    example = next(iter(TemplateGenerator().generate(doc)))
    assert example.messages[0].content == "Traduza.\n\ncódigo limpo"
    assert example.messages[-1].content == "clean code"


def test_jsonl_and_wrapped_json_and_semicolons() -> None:
    jsonl = (
        '{"pergunta": "a?", "resposta": "resposta um"}\n'
        '{"pergunta": "b?", "resposta": "resposta dois"}'
    )
    assert len(list(TemplateGenerator().generate(Document(jsonl, source="d.jsonl")))) == 2

    wrapped = json.dumps({"data": [{"question": "q?", "answer": "a"}]})
    assert len(list(TemplateGenerator().generate(Document(wrapped, source="d.json")))) == 1

    semicolons = Document("pergunta;resposta\nq1?;r1\nq2?;r2\n", source="d.csv")
    assert len(list(TemplateGenerator().generate(semicolons))) == 2


def test_system_column_overrides_the_default() -> None:
    rows = [
        {"question": "q1?", "answer": "a1", "system": "Do banco."},
        {"question": "q2?", "answer": "a2", "system": ""},
    ]
    doc = Document(json.dumps(rows), source="d.json")
    examples = list(TemplateGenerator(system="Padrão.").generate(doc))
    assert [e.system for e in examples] == ["Do banco.", "Padrão."]


def test_missing_column_and_empty_values_are_reported() -> None:
    doc = Document("nome\nCafé\n", source="p.csv")
    with pytest.raises(FormatError, match="missing column"):
        list(TemplateGenerator(user="{nome}?", assistant="{preco}").generate(doc))

    blanks = Document("pergunta,resposta\nq?,\n", source="p.csv")
    with pytest.raises(FormatError, match="empty example"):
        list(TemplateGenerator().generate(blanks))
    assert list(TemplateGenerator(skip_incomplete=True).generate(blanks)) == []


def test_unknown_columns_and_half_a_template_are_configuration_errors() -> None:
    doc = Document("foo,bar\n1,2\n", source="p.csv")
    with pytest.raises(ConfigError, match="could not detect"):
        list(TemplateGenerator().generate(doc))
    with pytest.raises(ConfigError, match="both"):
        TemplateGenerator(user="{a}")


def test_invalid_json_is_a_format_error() -> None:
    with pytest.raises(FormatError, match="invalid JSON"):
        list(TemplateGenerator().generate(Document("{oops}", source="d.jsonl")))
    with pytest.raises(FormatError, match="array of objects"):
        list(TemplateGenerator().generate(Document('["a"]', source="d.json")))


def test_empty_source_and_describe() -> None:
    assert list(TemplateGenerator().generate(Document("  ", source="p.csv"))) == []
    generator = TemplateGenerator(user="{a}", assistant="{b}", system="s")
    assert generator.name == "template"
    assert dict(generator.describe()) == {
        "user": "{a}",
        "assistant": "{b}",
        "system": "s",
        "skip_incomplete": False,
    }
