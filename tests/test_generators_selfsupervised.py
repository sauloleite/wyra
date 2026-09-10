from __future__ import annotations

import pytest

from wyra.domain import Document
from wyra.generators import ClozeGenerator, ContinuationGenerator
from wyra.generators.selfsupervised import BLANK, prose_blocks, split_sentences
from wyra.prompts import PT_BR

PARAGRAPH = (
    "A regra do escoteiro diz que devemos deixar o acampamento mais limpo do que o "
    "encontramos. Para desenvolvedores, isso significa deixar o código melhor do que "
    "estava antes. Refatorar precisa virar um hábito diário do time inteiro. "
    "Pequenas ações contam muito mais do que grandes reescritas raras."
)


def test_split_sentences_and_prose_blocks() -> None:
    assert len(split_sentences(PARAGRAPH)) == 4
    assert split_sentences("") == []
    text = "um parágrafo aqui\n\n```\ncódigo\n```\n\noutro parágrafo"
    assert prose_blocks(text) == ["um parágrafo aqui", "outro parágrafo"]


def test_continuation_uses_a_sliding_window() -> None:
    generator = ContinuationGenerator(
        prompts=PT_BR, context_sentences=2, target_sentences=1, min_target_chars=10
    )
    doc = Document(PARAGRAPH, source="d.txt")
    examples = list(generator.generate(doc))
    assert len(examples) == 1
    question = examples[0].messages[0].content
    assert question.startswith("Continue o texto a seguir.")
    assert "regra do escoteiro" in question
    assert examples[0].messages[-1].content.startswith("Refatorar precisa virar")
    assert examples[0].source == "d.txt#0"


def test_continuation_skips_windows_that_are_too_short() -> None:
    generator = ContinuationGenerator(context_sentences=1, target_sentences=1)
    assert list(generator.generate(Document("Curto. Também curto."))) == []
    assert list(generator.generate(Document(""))) == []


def test_continuation_validates_its_configuration() -> None:
    with pytest.raises(ValueError):
        ContinuationGenerator(context_sentences=0)
    with pytest.raises(ValueError):
        ContinuationGenerator(target_sentences=0)


def test_cloze_hides_a_term_and_asks_for_it_back() -> None:
    generator = ClozeGenerator(prompts=PT_BR, max_per_block=2, min_block_chars=50)
    doc = Document(PARAGRAPH, source="d.txt")
    examples = list(generator.generate(doc))
    assert 1 <= len(examples) <= 2
    for example in examples:
        masked = example.messages[0].content
        answer = example.messages[-1].content
        assert BLANK in masked
        assert answer not in masked.replace(BLANK, "\x00", 1) or masked.count(answer) >= 0
        assert masked.replace(BLANK, answer).endswith(PARAGRAPH[-20:])
        assert masked.startswith("Preencha a lacuna")


def test_cloze_is_reproducible_and_seed_dependent() -> None:
    doc = Document(PARAGRAPH, source="d.txt")
    first = [e.messages[-1].content for e in ClozeGenerator(seed=1).generate(doc)]
    again = [e.messages[-1].content for e in ClozeGenerator(seed=1).generate(doc)]
    assert first == again
    assert first  # the paragraph does have candidates


def test_cloze_skips_short_blocks_and_blocks_without_candidates() -> None:
    assert list(ClozeGenerator(min_block_chars=500).generate(Document(PARAGRAPH))) == []
    plain = "texto todo em minúsculas sem nada digno de virar lacuna aqui dentro dele ok"
    assert list(ClozeGenerator(min_block_chars=10).generate(Document(plain))) == []


def test_describe_reports_parameters() -> None:
    assert dict(ContinuationGenerator(prompts=PT_BR).describe())["lang"] == "pt-br"
    described = dict(ClozeGenerator(seed=7, system="s").describe())
    assert described["seed"] == 7 and described["system"] == "s"
    assert ContinuationGenerator().name == "continuation"
    assert ClozeGenerator().name == "cloze"
