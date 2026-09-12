"""The lazy SDK imports and the platform branches, exercised with stub modules.

These are the paths that break silently for somebody else: a user who installs an extra, or
runs on an operating system the author does not. A stub module makes them deterministic and
keeps the suite offline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from wyra.providers import modelstore as ms


def stub_module(name: str, **attributes: Any) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def test_the_embedded_engine_is_imported_once_and_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from wyra.providers.embedded import EmbeddedProvider

    engine = stub_module("onnxruntime_genai", __version__="9.9.9")
    monkeypatch.setitem(sys.modules, "onnxruntime_genai", engine)

    provider = EmbeddedProvider(model_dir=tmp_path)
    assert provider._engine() is engine
    monkeypatch.delitem(sys.modules, "onnxruntime_genai")
    assert provider._engine() is engine  # cached, so the second call does not re-import


def test_the_gemini_client_is_built_from_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from wyra.providers.gemini import GeminiProvider

    built: dict[str, Any] = {}

    def client(**kwargs: Any) -> SimpleNamespace:
        built.update(kwargs)
        return SimpleNamespace(models=None)

    genai = stub_module("google.genai", Client=client)
    monkeypatch.setitem(sys.modules, "google", stub_module("google", genai=genai))
    monkeypatch.setitem(sys.modules, "google.genai", genai)

    provider = GeminiProvider("gemini-2.5-flash", api_key="key-from-the-environment")
    assert provider.model == "gemini-2.5-flash"
    assert built == {"api_key": "key-from-the-environment"}


def test_the_openai_client_is_built_from_key_and_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    from wyra.providers.openai_compat import OpenAICompatibleProvider

    built: dict[str, Any] = {}

    def openai_client(**kwargs: Any) -> SimpleNamespace:
        built.update(kwargs)
        return SimpleNamespace(chat=None)

    monkeypatch.setitem(sys.modules, "openai", stub_module("openai", OpenAI=openai_client))

    OpenAICompatibleProvider("gpt-4o-mini", api_key="sk-test")
    assert built == {"api_key": "sk-test", "base_url": None}

    built.clear()
    OpenAICompatibleProvider("local-model", base_url="http://localhost:1234/v1")
    assert built == {"api_key": "not-needed", "base_url": "http://localhost:1234/v1"}


def test_the_tiktoken_counter_delegates_to_the_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    from wyra.tokens import TiktokenCounter

    encoding = SimpleNamespace(encode=lambda text: list(text.split()))
    monkeypatch.setitem(
        sys.modules, "tiktoken", stub_module("tiktoken", get_encoding=lambda name: encoding)
    )

    counter = TiktokenCounter("o200k_base")
    assert counter.name == "tiktoken:o200k_base"
    assert counter.count("um dois tres") == 3


@pytest.mark.parametrize(
    ("platform", "env", "expected_parts"),
    [
        ("win32", {"LOCALAPPDATA": "/appdata"}, ("/appdata", "wyra", "models")),
        ("win32", {}, ("AppData", "Local", "wyra", "models")),
        ("darwin", {}, ("Library", "Caches", "wyra", "models")),
        ("linux", {"XDG_CACHE_HOME": "/xdg"}, ("/xdg", "wyra", "models")),
        ("linux", {}, (".cache", "wyra", "models")),
    ],
)
def test_the_cache_lives_where_each_platform_expects(
    platform: str,
    env: dict[str, str],
    expected_parts: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    for name in ("WYRA_CACHE_DIR", "LOCALAPPDATA", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    root = ms.cache_root()
    assert root.is_absolute()
    for part in expected_parts:
        assert part in str(root), f"{part} missing from {root}"


def test_resolve_downloads_when_it_is_allowed_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: dict[str, Any] = {}

    def fake_fetch(entry: Any, **kwargs: Any) -> Path:
        called["name"] = entry.name
        target = ms.model_dir(entry, kwargs.get("cache_dir"))
        target.mkdir(parents=True, exist_ok=True)
        (target / ms.CONFIG_FILE).write_text("{}", encoding="utf-8")
        return target

    monkeypatch.setattr(ms, "fetch", fake_fetch)
    path = ms.resolve("qwen2.5-0.5b", cache_dir=tmp_path, download=True)
    assert called["name"] == "qwen2.5-0.5b"
    assert path == tmp_path / "qwen2.5-0.5b"


def test_the_default_opener_really_delegates_to_urlopen(tmp_path: Path) -> None:
    """A file:// URL proves the delegation without touching the network."""
    import urllib.request

    payload = b'{"siblings": []}'
    source = tmp_path / "payload.json"
    source.write_bytes(payload)

    request = urllib.request.Request(source.as_uri())
    with ms._default_opener(request, 5.0) as response:
        assert response.read() == payload
