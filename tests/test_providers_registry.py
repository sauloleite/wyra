from __future__ import annotations

import builtins

import pytest

from wyra.config import Settings
from wyra.errors import ConfigError
from wyra.providers import (
    PROVIDERS,
    FakeCompletionProvider,
    available,
    create,
    from_env,
    register,
)


def test_create_a_fake_and_an_ollama_provider() -> None:
    assert isinstance(create("fake"), FakeCompletionProvider)
    provider = create("ollama", model="llama3.2:1b", settings=Settings.from_env({}))
    assert provider.name == "ollama" and provider.model == "llama3.2:1b"


def test_the_environment_chooses_the_provider() -> None:
    settings = Settings.from_env({"WYRA_PROVIDER": "ollama", "WYRA_MODEL": "qwen3:4b"})
    assert create(settings=settings).model == "qwen3:4b"
    assert from_env({"WYRA_PROVIDER": "ollama", "OLLAMA_HOST": "box:1"}).name == "ollama"


def test_unknown_provider_lists_the_known_ones() -> None:
    with pytest.raises(ConfigError) as info:
        create("mistral")
    assert "unknown provider 'mistral'" in str(info.value)
    assert "ollama" in str(info.value)
    assert available() == ["fake", "gemini", "ollama", "openai"]


def test_a_missing_api_key_fails_before_any_network_call() -> None:
    settings = Settings.from_env({})
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        create("gemini", settings=settings)
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        create("openai", settings=settings)


def test_a_missing_extra_names_the_install_command(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name in ("openai", "google") or name.startswith("google."):
            raise ImportError(f"no {name}")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    settings = Settings.from_env({"OPENAI_API_KEY": "sk-x", "GEMINI_API_KEY": "g-x"})
    with pytest.raises(ConfigError, match=r"wyra\[openai\]"):
        create("openai", settings=settings)
    with pytest.raises(ConfigError, match=r"wyra\[gemini\]"):
        create("gemini", settings=settings)


def test_register_adds_a_provider() -> None:
    def echo(
        *, model: str | None = None, settings: Settings | None = None
    ) -> FakeCompletionProvider:
        return FakeCompletionProvider(name="echo", model=model or "echo-1")

    try:
        register("echo", echo)
        assert "echo" in available()
        provider = create("echo", settings=Settings.from_env({}), model="echo-2")
        assert (provider.name, provider.model) == ("echo", "echo-2")
    finally:
        PROVIDERS.pop("echo", None)


def test_a_local_openai_compatible_server_needs_no_key() -> None:
    class StubClient:
        chat = None

    settings = Settings.from_env({"OPENAI_BASE_URL": "http://localhost:1234/v1"})
    provider = create("openai", settings=settings, client=StubClient(), model="local")
    assert provider.model == "local"
