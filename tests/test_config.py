from __future__ import annotations

from wyra.config import DEFAULT_MODELS, DEFAULT_OLLAMA_HOST, Settings, normalize_host


def test_defaults_when_the_environment_is_empty() -> None:
    settings = Settings.from_env({})
    assert settings.provider == "ollama"
    assert settings.model is None
    assert settings.ollama_host == DEFAULT_OLLAMA_HOST
    assert settings.openai_api_key is None and settings.gemini_api_key is None
    assert settings.model_for("openai") == DEFAULT_MODELS["openai"]
    assert settings.model_for("unknown") == ""


def test_environment_values_are_read_and_trimmed() -> None:
    settings = Settings.from_env(
        {
            "WYRA_PROVIDER": " Gemini ",
            "WYRA_MODEL": " gemini-2.5-flash ",
            "OLLAMA_HOST": "box:11434/",
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_BASE_URL": "http://localhost:1234/v1",
            "GEMINI_API_KEY": "  ",
            "GOOGLE_API_KEY": "google-key",
        }
    )
    assert settings.provider == "gemini"
    assert settings.model == "gemini-2.5-flash"
    assert settings.ollama_host == "http://box:11434"
    assert settings.openai_api_key == "sk-test"
    assert settings.openai_base_url == "http://localhost:1234/v1"
    assert settings.gemini_api_key == "google-key"  # falls back when GEMINI_API_KEY is blank
    assert settings.model_for("gemini") == "gemini-2.5-flash"


def test_gemini_key_wins_over_the_google_one() -> None:
    settings = Settings.from_env({"GEMINI_API_KEY": "a", "GOOGLE_API_KEY": "b"})
    assert settings.gemini_api_key == "a"


def test_normalize_host() -> None:
    assert normalize_host("localhost:11434") == "http://localhost:11434"
    assert normalize_host("https://ollama.internal/") == "https://ollama.internal"
    assert normalize_host("  ") == DEFAULT_OLLAMA_HOST


def test_settings_read_the_real_environment_by_default(monkeypatch) -> None:
    monkeypatch.setenv("WYRA_PROVIDER", "openai")
    monkeypatch.delenv("WYRA_MODEL", raising=False)
    assert Settings.from_env().provider == "openai"
