from __future__ import annotations

import json

import pytest

from wyra.domain import Document, Role
from wyra.errors import GenerationError, ProviderError
from wyra.generators.llm import InstructionGenerator, QAPairGenerator, extract_json
from wyra.prompts import PT_BR
from wyra.providers import FakeCompletionProvider

TEXT = (
    "A regra do escoteiro diz para deixar o código mais limpo do que estava antes. "
    "Refatorar precisa virar um hábito diário do time inteiro."
)
DOC = Document(TEXT, source="cc.md", index=2)
PAIRS = json.dumps(
    {
        "pairs": [
            {
                "question": "O que diz a regra do escoteiro?",
                "answer": "Deixe o código mais limpo do que estava antes de mexer nele.",
            },
            {
                "question": "Refatorar é evento raro?",
                "answer": "Não, refatorar precisa virar um hábito diário do time.",
            },
        ]
    },
    ensure_ascii=False,
)


def test_extract_json_handles_fences_and_prose() -> None:
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Claro! Aqui está:\n{"a": 1}\nEspero ter ajudado.') == {"a": 1}
    assert extract_json("[1, 2]") == [1, 2]
    with pytest.raises(GenerationError, match="not JSON"):
        extract_json("desculpe, não posso ajudar")


def test_pairs_become_examples_with_lineage() -> None:
    provider = FakeCompletionProvider(responses=[PAIRS])
    generator = QAPairGenerator(provider, prompts=PT_BR, per_chunk=2, system="Seja fiel.")
    examples = list(generator.generate(DOC))

    assert len(examples) == 2
    assert examples[0].messages[0].role is Role.SYSTEM
    assert examples[0].system == "Seja fiel."
    assert all(e.source == "cc.md#2" for e in examples)
    assert examples[0].messages[1].content == "O que diz a regra do escoteiro?"


def test_the_request_carries_the_schema_temperature_and_language() -> None:
    provider = FakeCompletionProvider(responses=[PAIRS])
    QAPairGenerator(provider, prompts=PT_BR, per_chunk=4, temperature=0.7).generate(DOC).__next__()

    call = provider.calls[0]
    assert call.temperature == 0.7
    assert call.json_schema is not None
    assert call.json_schema["properties"]["pairs"]["items"]["required"] == ["question", "answer"]
    assert call.json_schema["additionalProperties"] is False
    assert call.messages[0].role is Role.SYSTEM
    assert "Você escreve dados de treinamento" in call.messages[0].content
    assert "escreva 4 pares" in call.messages[1].content
    assert TEXT in call.messages[1].content


def test_a_non_json_reply_is_retried_once_then_raises() -> None:
    provider = FakeCompletionProvider(responses=["desculpe", "ainda desculpe"])
    with pytest.raises(GenerationError, match="not JSON"):
        list(QAPairGenerator(provider, prompts=PT_BR).generate(DOC))

    assert len(provider.calls) == 2
    assert provider.calls[1].messages[-1].content == PT_BR.json_only
    assert provider.calls[1].messages[-2].content == "desculpe"


def test_a_retry_that_succeeds_yields_examples() -> None:
    provider = FakeCompletionProvider(responses=["desculpe", PAIRS])
    examples = list(QAPairGenerator(provider, prompts=PT_BR).generate(DOC))
    assert len(examples) == 2 and len(provider.calls) == 2


def test_retries_can_be_disabled() -> None:
    provider = FakeCompletionProvider(responses=["desculpe", PAIRS])
    with pytest.raises(GenerationError):
        list(QAPairGenerator(provider, max_retries=0).generate(DOC))
    assert len(provider.calls) == 1


def test_bad_shapes_are_generation_errors() -> None:
    for reply in ('{"pairs": "nope"}', '{"pairs": []}', '{"pairs": [1, 2]}', '{"other": []}'):
        with pytest.raises(GenerationError):
            list(
                QAPairGenerator(
                    FakeCompletionProvider(responses=[reply, reply]), max_retries=0
                ).generate(DOC)
            )


def test_a_bare_list_is_accepted_and_empty_fields_are_dropped() -> None:
    reply = json.dumps(
        [
            {"question": "O que diz a regra do escoteiro?", "answer": "Deixe o código mais limpo."},
            {"question": "", "answer": "sem pergunta"},
        ],
        ensure_ascii=False,
    )
    examples = list(QAPairGenerator(FakeCompletionProvider(responses=[reply])).generate(DOC))
    assert len(examples) == 1


def test_ungrounded_answers_are_dropped() -> None:
    reply = json.dumps(
        {
            "pairs": [
                {
                    "question": "Qual a capital da Austrália?",
                    "answer": "Camberra fica na Austrália.",
                },
                {
                    "question": "O que diz a regra?",
                    "answer": "Deixe o código mais limpo do que estava.",
                },
            ]
        },
        ensure_ascii=False,
    )
    kept = list(
        QAPairGenerator(FakeCompletionProvider(responses=[reply]), min_grounding=0.5).generate(DOC)
    )
    assert [e.messages[-1].content for e in kept] == ["Deixe o código mais limpo do que estava."]
    all_kept = list(
        QAPairGenerator(FakeCompletionProvider(responses=[reply]), min_grounding=0.0).generate(DOC)
    )
    assert len(all_kept) == 2


def test_provider_errors_propagate_untouched() -> None:
    provider = FakeCompletionProvider(responses=[ProviderError("connection refused")])
    with pytest.raises(ProviderError, match="connection refused"):
        list(QAPairGenerator(provider).generate(DOC))


def test_truncated_output_is_reported_as_such() -> None:
    provider = FakeCompletionProvider(responses=[PAIRS], finish_reason="length")
    with pytest.raises(GenerationError, match="truncated"):
        list(QAPairGenerator(provider).generate(DOC))


def test_oversized_chunks_are_refused_before_the_call() -> None:
    provider = FakeCompletionProvider(responses=[PAIRS])
    huge = Document("palavra " * 5000, source="big.txt")
    with pytest.raises(GenerationError, match="max_input_tokens"):
        list(QAPairGenerator(provider, max_input_tokens=100).generate(huge))
    assert provider.calls == []


def test_empty_documents_never_reach_the_provider() -> None:
    provider = FakeCompletionProvider(responses=[PAIRS])
    assert list(QAPairGenerator(provider).generate(Document("  "))) == []
    assert provider.calls == []


def test_instruction_generator_merges_input_into_the_question() -> None:
    reply = json.dumps(
        {
            "items": [
                {
                    "instruction": "Explique a regra do escoteiro.",
                    "input": "Em uma frase.",
                    "output": "Deixe o código mais limpo do que estava antes.",
                },
                {"instruction": "Sem saída.", "input": "", "output": ""},
            ]
        },
        ensure_ascii=False,
    )
    examples = list(
        InstructionGenerator(FakeCompletionProvider(responses=[reply]), prompts=PT_BR).generate(DOC)
    )
    assert len(examples) == 1
    assert examples[0].messages[0].content == "Explique a regra do escoteiro.\n\nEm uma frase."
    schema = InstructionGenerator(FakeCompletionProvider()).response_schema()
    assert schema["properties"]["items"]["items"]["required"] == ["instruction", "input", "output"]


def test_describe_pins_the_prompt_version() -> None:
    generator = QAPairGenerator(FakeCompletionProvider(), prompts=PT_BR, per_chunk=5)
    described = dict(generator.describe())
    assert described["provider"] == "fake" and described["model"] == "fake-model"
    assert described["per_chunk"] == 5 and described["lang"] == "pt-br"
    assert len(described["prompt_sha256"]) == 64
    other = dict(QAPairGenerator(FakeCompletionProvider(), per_chunk=5).describe())
    assert other["prompt_sha256"] != described["prompt_sha256"]
    assert generator.name == "llm-qa"


def test_fake_provider_helpers() -> None:
    provider = FakeCompletionProvider(responses=[lambda messages: f"{len(messages)} msgs"])
    completion = provider.complete([], temperature=0.3)
    assert completion.text == "0 msgs" and completion.model == "fake-model"
    assert provider.calls[0].prompt == ""
    with pytest.raises(ProviderError, match="no responses"):
        FakeCompletionProvider(responses=[]).complete([])


class ProviderWithLineage(FakeCompletionProvider):
    def lineage(self) -> dict[str, str]:
        return {"repo": "microsoft/Phi-3.5-mini-instruct-onnx", "revision": "7230dcd"}


class ProviderWithBrokenLineage(FakeCompletionProvider):
    def lineage(self) -> dict[str, str]:
        raise RuntimeError("the cache is gone")


class ProviderWithOddLineage(FakeCompletionProvider):
    def lineage(self) -> list[str]:
        return ["not", "a", "mapping"]


def test_describe_pins_the_weights_when_the_provider_knows_them() -> None:
    described = dict(QAPairGenerator(ProviderWithLineage(responses=[PAIRS])).describe())
    assert described["weights"] == {
        "repo": "microsoft/Phi-3.5-mini-instruct-onnx",
        "revision": "7230dcd",
    }


def test_describe_omits_weights_when_there_is_no_provenance() -> None:
    assert "weights" not in dict(QAPairGenerator(FakeCompletionProvider()).describe())


def test_broken_or_odd_provenance_never_breaks_a_build() -> None:
    assert "weights" not in dict(QAPairGenerator(ProviderWithBrokenLineage()).describe())
    assert "weights" not in dict(QAPairGenerator(ProviderWithOddLineage()).describe())
