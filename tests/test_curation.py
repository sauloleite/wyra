from __future__ import annotations

from wyra.curation import (
    DEFAULT_STEPS,
    Dedup,
    MinLength,
    NearDuplicate,
    Normalize,
    TokenBudget,
    run_pipeline,
)
from wyra.domain import Example, Message, Role


def test_normalize_cleans_whitespace_and_unicode_form() -> None:
    messy = Example.qa("  á quest̃ion  \r\n\r\n\r\n more \t\n", "answer   ")
    clean = next(iter(Normalize().apply([messy])))
    # leading indentation is preserved on purpose (Markdown lists, indented code)
    assert clean.messages[0].content == "á quest̃ion\n\n more"
    assert clean.messages[1].content == "answer"
    already = Example.qa("q", "a")
    assert next(iter(Normalize().apply([already]))) is already


def test_normalize_keeps_source_and_metadata() -> None:
    example = Example.qa("q ", "a ", system="s ", source="doc#1")
    clean = next(iter(Normalize().apply([example])))
    assert clean.source == "doc#1" and clean.system == "s"


def test_dedup_keeps_the_first_occurrence() -> None:
    first = Example.qa("O que é DRY?", "Primeira resposta.")
    same = Example.qa("o que é  dry?", "primeira resposta.")
    other = Example.qa("O que é DRY?", "Segunda resposta.")
    kept = list(Dedup().apply([first, same, other]))
    assert kept == [first, other]
    by_user = list(Dedup(key="user").apply([first, same, other]))
    assert by_user == [first]


def test_min_length_and_token_budget() -> None:
    short = Example.qa("q", "ok")
    long = Example.qa("q", "a" * 50)
    assert list(MinLength(min_chars=20).apply([short, long])) == [long]
    assert list(TokenBudget(max_tokens=1000).apply([short, long])) == [short, long]
    assert list(TokenBudget(max_tokens=20).apply([short, long])) == [short]


def test_near_duplicate_collapses_reworded_examples() -> None:
    base = (
        "A regra do escoteiro diz que você deve deixar o código mais limpo do que estava "
        "antes de você mexer nele, e refatorar precisa virar um hábito diário."
    )
    reworded = base.replace("diário", "constante")
    unrelated = (
        "Testes limpos seguem o acrônimo FIRST, que reúne rapidez, independência, "
        "repetibilidade, auto validação e pontualidade na escrita dos testes."
    )
    step = NearDuplicate(threshold=0.7)
    kept = list(
        step.apply(
            [Example.qa("q1?", base), Example.qa("q2?", reworded), Example.qa("q3?", unrelated)]
        )
    )
    assert [e.messages[-1].content for e in kept] == [base, unrelated]


def test_near_duplicate_passes_through_texts_shorter_than_the_shingle() -> None:
    tiny = [Example.qa("a", "b"), Example.qa("a", "b c")]
    assert list(NearDuplicate(shingle=5).apply(tiny)) == tiny


def test_run_pipeline_counts_each_step() -> None:
    examples = [
        Example.qa("q", "resposta longa o bastante"),
        Example.qa("q", "resposta longa o bastante"),
        Example.qa("outra?", "curta"),
    ]
    stream, report = run_pipeline(examples, [Normalize(), Dedup(), MinLength(min_chars=10)])
    assert report.generated == 0  # lazy: nothing consumed yet
    kept = list(stream)
    assert len(kept) == 1
    assert report.as_list() == [
        {"step": "normalize", "in": 3, "out": 3},
        {"step": "dedup", "in": 3, "out": 2},
        {"step": "min_length", "in": 2, "out": 1},
    ]
    assert report.generated == 3 and report.kept == 1
    assert report.steps[1].dropped == 1
    assert "dedup: 3/2" in report.summary()


def test_run_pipeline_with_no_steps_and_no_examples() -> None:
    stream, report = run_pipeline([], [])
    assert list(stream) == []
    assert report.generated == 0 and report.kept == 0 and report.summary() == ""
    stream, report = run_pipeline([Example.qa("q", "a")], DEFAULT_STEPS)
    assert len(list(stream)) == 1


def test_invalid_step_configuration() -> None:
    import pytest

    with pytest.raises(ValueError):
        Dedup(key="whatever")
    with pytest.raises(ValueError):
        NearDuplicate(threshold=0)
    with pytest.raises(ValueError):
        NearDuplicate(num_perm=10, bands=3)


def test_normalize_preserves_name_and_weight() -> None:
    example = Example(
        (Message(Role.USER, "q ", name="ana"), Message(Role.ASSISTANT, "a ", weight=1))
    )
    clean = next(iter(Normalize().apply([example])))
    assert clean.messages[0].name == "ana" and clean.messages[1].weight == 1
