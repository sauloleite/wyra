from __future__ import annotations

from pathlib import Path

from wyra.domain import Document
from wyra.generators import QAPairExtractor


def test_markers_in_two_languages(fixtures: Path) -> None:
    doc = Document((fixtures / "faq.txt").read_text(encoding="utf-8"), source="faq.txt")
    examples = list(QAPairExtractor().generate(doc))
    questions = [e.messages[0].content for e in examples]
    assert questions == [
        "O que é código limpo?",
        "Por que evitar repetição?",
        "What does FIRST stand for?",
    ]
    assert examples[0].messages[-1].content.startswith("É um código que qualquer")
    assert all(e.source == "faq.txt#0" for e in examples)


def test_multi_block_answers_are_joined() -> None:
    doc = Document(
        "Pergunta: Por que evitar repetição?\n\n"
        "Resposta: Porque manter duas cópias dobra o custo de cada mudança.\n\n"
        "Cada parte do sistema deve ter uma única representação.\n\n"
        "E um terceiro bloco que não deve entrar."
    )
    answer = next(iter(QAPairExtractor(max_answer_blocks=2).generate(doc))).messages[-1].content
    assert "única representação" in answer
    assert "terceiro bloco" not in answer


def test_question_without_an_answer_is_dropped() -> None:
    doc = Document("Q: sem resposta?\n\nQ: com resposta?\n\nA: esta resposta é longa o bastante.")
    examples = list(QAPairExtractor().generate(doc))
    assert [e.messages[0].content for e in examples] == ["com resposta?"]


def test_short_answers_are_dropped() -> None:
    doc = Document("Q: pergunta?\n\nA: curta.")
    assert list(QAPairExtractor(min_answer_chars=50).generate(doc)) == []


def test_prose_questions_are_used_when_no_markers_exist(fixtures: Path) -> None:
    doc = Document((fixtures / "clean_code.md").read_text(encoding="utf-8"), source="cc.md")
    examples = list(QAPairExtractor().generate(doc))
    questions = [e.messages[0].content for e in examples]
    assert "Por que isso acontece?" in questions
    answer = next(e for e in examples if e.messages[0].content == "Por que isso acontece?")
    assert answer.messages[-1].content.startswith("A gente sempre tem alguma desculpa")


def test_prose_questions_can_be_disabled_and_consecutive_questions_skipped() -> None:
    doc = Document(
        "Uma pergunta em prosa?\n\nOutra pergunta seguida?\n\nA resposta final vem aqui."
    )
    assert list(QAPairExtractor(prose_questions=False).generate(doc)) == []
    examples = list(QAPairExtractor().generate(doc))
    assert [e.messages[0].content for e in examples] == ["Outra pergunta seguida?"]


def test_lead_in_line_before_the_question_is_ignored() -> None:
    doc = Document(
        "Muitos devem se fazer a pergunta abaixo:\nMas vale o trabalho?\n\n"
        "A resposta é simples: o custo de manter código ruim é enorme."
    )
    example = next(iter(QAPairExtractor().generate(doc)))
    assert example.messages[0].content == "Mas vale o trabalho?"


def test_enumerated_questions_lose_their_bullet() -> None:
    doc = Document("1. O que é DRY?\n\nNão repita a si mesmo, em nenhuma parte do sistema.")
    example = next(iter(QAPairExtractor().generate(doc)))
    assert example.messages[0].content == "O que é DRY?"


def test_describe_and_system_prompt() -> None:
    generator = QAPairExtractor(system="Seja direto.")
    assert generator.name == "qa-pairs"
    described = dict(generator.describe())
    assert described["system"] == "Seja direto."
    assert ["pergunta", "resposta"] in described["markers"]
    doc = Document("Q: pergunta?\n\nA: uma resposta suficientemente longa.")
    assert next(iter(generator.generate(doc))).system == "Seja direto."


def test_empty_document() -> None:
    assert list(QAPairExtractor().generate(Document(""))) == []


def test_a_question_spread_over_two_lines_is_joined() -> None:
    doc = Document("Pergunta: O que é DRY\ne por que ele importa?\nResposta: Uma representação só.")
    example = next(iter(QAPairExtractor(min_answer_chars=5).generate(doc)))
    assert example.messages[0].content == "O que é DRY e por que ele importa?"


def test_max_answer_blocks_must_be_at_least_one() -> None:
    import pytest

    with pytest.raises(ValueError, match="max_answer_blocks"):
        QAPairExtractor(max_answer_blocks=0)
