"""Configuration from the environment. No file is read, nothing is hardcoded.

Twelve-Factor: every knob comes from the process environment, and a missing credential
fails fast with a message that says what to export, before any network call happens.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

DEFAULT_PROVIDER = "ollama"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# Sensible starting points, all overridable with WYRA_MODEL.
DEFAULT_MODELS: dict[str, str] = {
    "ollama": "llama3.2:3b",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
}


@dataclass(frozen=True, slots=True)
class Settings:
    provider: str = DEFAULT_PROVIDER
    model: str | None = None
    ollama_host: str = DEFAULT_OLLAMA_HOST
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    gemini_api_key: str | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        source = os.environ if env is None else env
        return cls(
            provider=(source.get("WYRA_PROVIDER") or DEFAULT_PROVIDER).strip().lower(),
            model=_clean(source.get("WYRA_MODEL")),
            ollama_host=normalize_host(source.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST),
            openai_api_key=_clean(source.get("OPENAI_API_KEY")),
            openai_base_url=_clean(source.get("OPENAI_BASE_URL")),
            gemini_api_key=_clean(source.get("GEMINI_API_KEY"))
            or _clean(source.get("GOOGLE_API_KEY")),
        )

    def model_for(self, provider: str) -> str:
        return self.model or DEFAULT_MODELS.get(provider, "")


def normalize_host(host: str) -> str:
    """``localhost:11434`` and ``http://localhost:11434/`` both mean the same server."""
    host = host.strip().rstrip("/")
    if not host:
        return DEFAULT_OLLAMA_HOST
    if "://" not in host:
        host = f"http://{host}"
    return host


def _clean(value: str | None) -> str | None:
    return value.strip() if value and value.strip() else None
