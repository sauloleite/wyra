from __future__ import annotations

import json
from pathlib import Path

import pytest

from wyra import build_dataset, build_examples, convert_jsonl
from wyra.chunking import ParagraphChunker
from wyra.domain import Document, Example
from wyra.errors import ConfigError, ValidationError
from wyra.generators import QAPairGenerator
from wyra.providers import FakeCompletionProvider

PAIRS = json.dumps(
    {
        "pairs": [
            {
                "question": "O que diz a regra do escoteiro?",
                "answer": "Deixe o código mais limpo do que estava antes de mexer nele.",
            }
        ]
    },
    ensure_ascii=False,
)


def test_markdown_to_jsonl_without_any_model(fixtures: Path, tmp_path: Path) -> None:
    result = build_dataset(
        fixtures / "clean_code.md",
        tmp_path,
        lang="pt-br",
        system_prompt="Você é um tutor de Clean Code.",
        validation_fraction=0.2,
    )
    assert result.train_path is not None and result.validation_path is not None
    records = [
        json.loads(line) for line in result.train_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records
    assert records[0]["messages"][0] == {
        "role": "system",
        "content": "Você é um tutor de Clean Code.",
    }
    assert any("Explique:" in r["messages"][1]["content"] for r in records)
    assert result.manifest.generator["name"] == "markdown-sections"
    assert result.manifest.chunker["name"] == "none"
    assert result.manifest.sources[0]["sha256"]
    assert result.counts["validation"] >= 1


def test_csv_and_json_use_the_template_generator(fixtures: Path, tmp_path: Path) -> None:
    result = build_dataset(fixtures / "qa.csv", tmp_path)
    assert result.counts["kept"] == 3
    assert result.manifest.generator["name"] == "template"

    explicit = build_dataset(
        Document("nome,preco\nCafé,10\n", source="p.csv"),
        tmp_path / "tpl",
        generator="template",
        user="Qual o preço de {nome}?",
        assistant="{nome} custa R$ {preco}.",
    )
    assert explicit.train[0].messages[0].content == "Qual o preço de Café?"


def test_prose_with_a_fake_provider_chunks_and_generates(tmp_path: Path) -> None:
    provider = FakeCompletionProvider(responses=[PAIRS])
    text = "\n\n".join(
        [
            "A regra do escoteiro diz para deixar o código mais limpo do que estava antes.",
            "Refatorar precisa virar um hábito diário do time inteiro que mantém o sistema.",
        ]
    )
    result = build_dataset(
        Document(text, source="notas.txt"),
        tmp_path,
        generator=QAPairGenerator(provider, per_chunk=1),
    )
    assert result.manifest.chunker["name"] == "paragraphs"
    assert result.manifest.generator["params"]["provider"] == "fake"
    assert result.counts["kept"] == 1
    # both paragraphs fit the default chunk budget, so one chunk means one call
    assert len(provider.calls) == 1
    assert "hábito diário" in provider.calls[0].messages[-1].content

    smaller = FakeCompletionProvider(responses=[PAIRS])
    build_dataset(
        Document(text, source="notas.txt"),
        tmp_path / "small",
        generator=QAPairGenerator(smaller, per_chunk=1),
        chunker=ParagraphChunker(max_chars=90, min_chars=10),
    )
    assert len(smaller.calls) == 2


def test_the_provider_can_be_named_and_resolved(tmp_path: Path) -> None:
    result = build_dataset(
        Document("texto qualquer para gerar", source="n.txt"),
        tmp_path,
        generator="llm-qa",
        provider="fake",
        lang="pt-br",
    )
    assert result.counts["generated"] == 0  # the default fake replies "{}", nothing usable
    assert result.counts["errors"] == 1


def test_auto_refuses_to_guess_for_plain_text_and_empty_sources(tmp_path: Path) -> None:
    note = tmp_path / "note.txt"
    note.write_text("prosa qualquer", encoding="utf-8")
    with pytest.raises(ConfigError, match="cannot guess"):
        build_dataset(note, tmp_path / "out")
    with pytest.raises(ConfigError, match="no sources"):
        build_dataset([], tmp_path / "out")


def test_options_reach_the_builder(fixtures: Path, tmp_path: Path) -> None:
    result = build_dataset(
        fixtures / "qa.csv",
        tmp_path,
        output_format="alpaca",
        max_tokens=5,
        dedupe=False,
        write_lineage=True,
    )
    assert result.manifest.output["format"] == "alpaca"
    assert [s["step"] for s in result.manifest.curation] == ["normalize", "token_budget"]
    assert result.counts["kept"] == 0  # every example is above a five-token budget
    assert (tmp_path / "lineage.jsonl").exists()


def test_convert_between_formats(fixtures: Path, tmp_path: Path) -> None:
    destination = tmp_path / "converted.jsonl"
    result = convert_jsonl(fixtures / "sharegpt.jsonl", destination)
    assert result.counts["kept"] == 2
    records = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert records[0]["messages"][0]["role"] == "system"
    assert destination.with_suffix(".jsonl.manifest.json").exists()

    to_alpaca = tmp_path / "alpaca.jsonl"
    convert_jsonl(fixtures / "good.jsonl", to_alpaca, output_format="alpaca")
    first = json.loads(to_alpaca.read_text(encoding="utf-8").splitlines()[0])
    assert set(first) == {"instruction", "input", "output", "system"}


def test_convert_deduplicates_and_can_split(fixtures: Path, tmp_path: Path) -> None:
    duplicated = tmp_path / "dup.jsonl"
    line = json.dumps(
        {"messages": [{"role": "user", "content": "q?"}, {"role": "assistant", "content": "a"}]}
    )
    duplicated.write_text(f"{line}\n{line}\n", encoding="utf-8")
    result = convert_jsonl(duplicated, tmp_path / "clean.jsonl")
    assert result.counts == {"generated": 2, "kept": 1, "train": 1, "validation": 0, "errors": 0}


def test_build_examples_writes_a_validation_file(tmp_path: Path) -> None:
    examples = [Example.qa(f"pergunta {i}?", f"resposta {i}") for i in range(10)]
    result = build_examples(examples, tmp_path / "data.jsonl", validation_fraction=0.2)
    assert result.validation_path is not None
    assert result.validation_path.name == "data.validation.jsonl"
    assert result.counts["train"] == 8 and result.counts["validation"] == 2
    assert result.manifest.generator["name"] == "prebuilt"


def test_the_default_chunk_size_suits_the_weakest_catalogue_model() -> None:
    from wyra.pipeline import DEFAULT_CHUNK_CHARS, _default_chunker
    from wyra.providers import FakeCompletionProvider

    # measured: the 0.5B model works at 651 source characters and degenerates at 910
    assert DEFAULT_CHUNK_CHARS <= 650

    chunker = _default_chunker(QAPairGenerator(FakeCompletionProvider()))
    assert chunker.name == "paragraphs"
    long_text = "\n\n".join(["parágrafo com algum conteúdo de verdade aqui." * 4] * 6)
    chunks = list(chunker.chunk(Document(long_text, source="p.txt")))
    assert len(chunks) > 1
    assert all(len(chunk.text) <= DEFAULT_CHUNK_CHARS for chunk in chunks)


def test_convert_can_skip_invalid_records(fixtures: Path, tmp_path: Path) -> None:
    destination = tmp_path / "salvaged.jsonl"
    result = convert_jsonl(fixtures / "bad.jsonl", destination, on_invalid="skip")
    assert result.counts["kept"] == 1
    assert destination.read_text(encoding="utf-8").count("\n") == 1

    with pytest.raises(ValidationError):
        convert_jsonl(fixtures / "bad.jsonl", tmp_path / "nope.jsonl")


def test_a_compressed_dataset_converts(fixtures: Path, tmp_path: Path) -> None:
    import gzip

    packed = tmp_path / "sharegpt.jsonl.gz"
    packed.write_bytes(gzip.compress((fixtures / "sharegpt.jsonl").read_bytes()))
    result = convert_jsonl(packed, tmp_path / "out.jsonl")
    assert result.counts["kept"] == 2


def test_a_provider_instance_can_be_passed_straight_in(tmp_path: Path) -> None:
    provider = FakeCompletionProvider(responses=[PAIRS])
    result = build_dataset(
        Document("Um texto sobre a regra do escoteuro e o código limpo.", source="n.txt"),
        tmp_path,
        generator="llm-qa",
        provider=provider,
        lang="pt-br",
    )
    assert result.counts["kept"] == 1
    assert len(provider.calls) == 1
    assert result.manifest.generator["params"]["provider"] == "fake"


def test_the_token_counter_reaches_the_manifest_and_the_budget(
    fixtures: Path, tmp_path: Path
) -> None:
    class CountsWords:
        name = "words"

        def count(self, text: str) -> int:
            return len(text.split())

    result = build_dataset(fixtures / "qa.csv", tmp_path, token_counter=CountsWords())
    assert result.manifest.stats["token_counter"] == "words"

    # the same counter must drive the budget filter, not a second default one
    tight = build_dataset(
        fixtures / "qa.csv", tmp_path / "tight", token_counter=CountsWords(), max_tokens=3
    )
    assert tight.counts["kept"] < result.counts["kept"]
    assert [s["step"] for s in tight.manifest.curation][-1] == "token_budget"


def test_an_unknown_counter_name_is_a_configuration_error(fixtures: Path, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown token counter"):
        build_dataset(fixtures / "qa.csv", tmp_path, token_counter="magic")


def test_build_examples_and_convert_accept_a_counter(fixtures: Path, tmp_path: Path) -> None:
    result = build_examples([Example.qa("q?", "a")], tmp_path / "d.jsonl", token_counter="approx")
    assert result.manifest.stats["token_counter"] == "approx"
    converted = convert_jsonl(fixtures / "good.jsonl", tmp_path / "c.jsonl", token_counter="approx")
    assert converted.manifest.stats["token_counter"] == "approx"


def test_an_existing_chat_dataset_is_pointed_at_convert(fixtures: Path, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="wyra convert"):
        build_dataset(fixtures / "good.jsonl", tmp_path)
    with pytest.raises(ConfigError, match="wyra convert"):
        build_dataset(fixtures / "sharegpt.jsonl", tmp_path)
    # an alpaca table is a table, and the tabular generator handles it
    assert build_dataset(fixtures / "alpaca.jsonl", tmp_path / "alp").counts["kept"] == 2


def test_the_extension_contract_is_importable_from_the_package() -> None:
    import wyra

    for name in (
        "ExampleGenerator",
        "CompletionProvider",
        "Chunker",
        "TokenCounter",
        "CurationStep",
        "DatasetWriter",
    ):
        assert hasattr(wyra, name), name
        assert name in wyra.__all__
