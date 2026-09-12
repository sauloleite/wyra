"""EmbeddedProvider against a stub engine: no onnxruntime-genai, no weights, no network."""

from __future__ import annotations

import builtins
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from wyra.domain import Message, Role
from wyra.errors import ConfigError, ProviderError
from wyra.providers.embedded import EmbeddedProvider

MESSAGES = [Message(Role.SYSTEM, "seja fiel"), Message(Role.USER, "gere pares")]
SCHEMA: dict[str, Any] = {"type": "object", "properties": {"pairs": {"type": "array"}}}
REPLY = '{"pairs": []}'


class StubTokenizer:
    def __init__(self, model: Any, *, template: bool = True) -> None:
        self.model = model
        self.template = template
        self.templated: Any = None
        self.decoded: list[list[int]] = []

    def apply_chat_template(self, *, messages: str, add_generation_prompt: bool = True) -> str:
        if not self.template:
            raise RuntimeError("this model has no chat template")
        self.templated = json.loads(messages)
        self.add_generation_prompt = add_generation_prompt
        return "RENDERED PROMPT"

    def encode(self, text: str) -> list[int]:
        self.encoded = text
        return [1, 2, 3]

    def decode(self, tokens: Any) -> str:
        self.decoded.append(list(tokens))
        return REPLY


class StubParams:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.search: dict[str, Any] = {}
        self.guidance: tuple[str, Any] | None = None

    def set_search_options(self, **kwargs: Any) -> None:
        self.search.update(kwargs)

    def set_guidance(self, kind: str, data: str, enable_ff_tokens: bool = False) -> None:
        self.guidance = (kind, json.loads(data))


class StubGenerator:
    def __init__(self, model: Any, params: StubParams) -> None:
        self.params = params
        self.appended: list[int] = []
        self.steps = 0

    def append_tokens(self, tokens: Any) -> None:
        self.appended = list(tokens)

    def generate_next_token(self) -> None:
        self.steps += 1

    def is_done(self) -> bool:
        return self.steps >= 2

    def get_sequence(self, index: int) -> list[int]:
        return [1, 2, 3, 10, 11]


def make_engine(
    *, template: bool = True, fail_load: bool = False, fail_generate: bool = False
) -> Any:
    state: dict[str, Any] = {"models": 0, "params": [], "generators": []}

    def model_factory(path: str) -> Any:
        if fail_load:
            raise RuntimeError("corrupt genai_config.json")
        state["models"] += 1
        state["path"] = path
        return SimpleNamespace(path=path)

    def tokenizer_factory(model: Any) -> StubTokenizer:
        tokenizer = StubTokenizer(model, template=template)
        state["tokenizer"] = tokenizer
        return tokenizer

    def params_factory(model: Any) -> StubParams:
        params = StubParams(model)
        state["params"].append(params)
        return params

    def generator_factory(model: Any, params: StubParams) -> StubGenerator:
        if fail_generate:
            raise RuntimeError("out of memory")
        generator = StubGenerator(model, params)
        state["generators"].append(generator)
        return generator

    return SimpleNamespace(
        Model=model_factory,
        Tokenizer=tokenizer_factory,
        GeneratorParams=params_factory,
        Generator=generator_factory,
        state=state,
    )


def test_schema_reaches_constrained_decoding(tmp_path: Path) -> None:
    engine = make_engine()
    provider = EmbeddedProvider(model_dir=tmp_path, runtime=engine, max_new_tokens=64)
    completion = provider.complete(MESSAGES, json_schema=SCHEMA, temperature=0.0)

    assert provider.name == "local"
    params = engine.state["params"][0]
    assert params.guidance == ("json_schema", SCHEMA)
    assert params.search == {"max_length": 3 + 64, "do_sample": False, "temperature": 0.0}
    assert completion.text == REPLY
    assert (completion.input_tokens, completion.output_tokens) == (3, 2)
    assert completion.finish_reason == "stop"
    assert completion.model == "phi-3.5-mini"


def test_only_the_generated_tokens_are_decoded(tmp_path: Path) -> None:
    engine = make_engine()
    EmbeddedProvider(model_dir=tmp_path, runtime=engine).complete(MESSAGES)
    assert engine.state["tokenizer"].decoded == [[10, 11]]
    assert engine.state["generators"][0].appended == [1, 2, 3]


def test_sampling_is_enabled_only_above_zero_temperature(tmp_path: Path) -> None:
    engine = make_engine()
    provider = EmbeddedProvider(model_dir=tmp_path, runtime=engine)
    provider.complete(MESSAGES, temperature=0.7)
    assert engine.state["params"][0].search["do_sample"] is True
    assert engine.state["params"][0].search["temperature"] == 0.7


def test_no_schema_means_no_guidance(tmp_path: Path) -> None:
    engine = make_engine()
    EmbeddedProvider(model_dir=tmp_path, runtime=engine).complete(MESSAGES)
    assert engine.state["params"][0].guidance is None


def test_the_chat_template_is_used_when_the_model_has_one(tmp_path: Path) -> None:
    engine = make_engine()
    EmbeddedProvider(model_dir=tmp_path, runtime=engine).complete(MESSAGES)
    tokenizer = engine.state["tokenizer"]
    assert tokenizer.templated == [
        {"role": "system", "content": "seja fiel"},
        {"role": "user", "content": "gere pares"},
    ]
    assert tokenizer.encoded == "RENDERED PROMPT"


def test_a_plain_transcript_is_the_fallback(tmp_path: Path) -> None:
    engine = make_engine(template=False)
    EmbeddedProvider(model_dir=tmp_path, runtime=engine).complete(MESSAGES)
    prompt = engine.state["tokenizer"].encoded
    assert prompt.startswith("system: seja fiel")
    assert prompt.endswith("assistant:")


def test_truncation_is_reported(tmp_path: Path) -> None:
    engine = make_engine()
    provider = EmbeddedProvider(model_dir=tmp_path, runtime=engine, max_new_tokens=2)
    assert provider.complete(MESSAGES).finish_reason == "length"


def test_the_model_is_loaded_once_and_reused(tmp_path: Path) -> None:
    engine = make_engine()
    provider = EmbeddedProvider(model_dir=tmp_path, runtime=engine)
    provider.complete(MESSAGES)
    provider.complete(MESSAGES)
    assert engine.state["models"] == 1
    assert len(engine.state["generators"]) == 2


def test_engine_failures_become_provider_errors(tmp_path: Path) -> None:
    with pytest.raises(ProviderError, match="cannot load the local model"):
        EmbeddedProvider(model_dir=tmp_path, runtime=make_engine(fail_load=True)).complete(MESSAGES)
    with pytest.raises(ProviderError, match="local generation failed"):
        EmbeddedProvider(model_dir=tmp_path, runtime=make_engine(fail_generate=True)).complete(
            MESSAGES
        )


def test_a_missing_extra_names_the_install_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "onnxruntime_genai":
            raise ImportError("no engine")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ConfigError, match=r"wyra\[local\]"):
        EmbeddedProvider(model_dir=tmp_path).complete(MESSAGES)


def test_an_explicit_directory_skips_the_catalogue(tmp_path: Path) -> None:
    provider = EmbeddedProvider(model_dir=tmp_path, runtime=make_engine())
    assert provider.model_path() == tmp_path


def test_the_registry_builds_it(tmp_path: Path) -> None:
    from wyra.config import Settings
    from wyra.providers import available, create

    assert "local" in available()
    provider = create(
        "local",
        settings=Settings.from_env({}),
        model=None,
        model_dir=tmp_path,
        runtime=make_engine(),
    )
    assert provider.name == "local"
