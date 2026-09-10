"""Provider registry: choose an adapter by name, import its SDK only when asked.

Lazy imports keep the core dependency-free and turn a missing extra into a
``ConfigError`` that names the install command instead of an ImportError traceback.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..config import DEFAULT_MODELS, Settings
from ..errors import ConfigError
from ..ports import CompletionProvider
from .fake import FakeCompletionProvider, RecordedCall

ProviderFactory = Callable[..., CompletionProvider]


def _ollama(**kwargs: Any) -> CompletionProvider:
    from .ollama import OllamaProvider

    return OllamaProvider(**kwargs)


def _openai(**kwargs: Any) -> CompletionProvider:
    from .openai_compat import OpenAICompatibleProvider

    return OpenAICompatibleProvider(**kwargs)


def _gemini(**kwargs: Any) -> CompletionProvider:
    from .gemini import GeminiProvider

    return GeminiProvider(**kwargs)


def _fake(**kwargs: Any) -> CompletionProvider:
    return FakeCompletionProvider(**kwargs)


PROVIDERS: dict[str, ProviderFactory] = {
    "ollama": _ollama,
    "openai": _openai,
    "gemini": _gemini,
    "fake": _fake,
}


def register(name: str, factory: ProviderFactory) -> None:
    """Add a provider, so ``--provider <name>`` and ``WYRA_PROVIDER`` find it.

    ``create`` calls the factory with ``model`` and ``settings`` keywords, so a custom
    factory must accept both (``**kwargs`` is enough).
    """
    PROVIDERS[name] = factory


def available() -> list[str]:
    return sorted(PROVIDERS)


def create(
    name: str | None = None,
    *,
    model: str | None = None,
    settings: Settings | None = None,
    **kwargs: Any,
) -> CompletionProvider:
    """Build a provider by name, falling back to WYRA_PROVIDER when none is given."""
    settings = settings or Settings.from_env()
    chosen = (name or settings.provider).strip().lower()
    try:
        factory = PROVIDERS[chosen]
    except KeyError:
        raise ConfigError(
            f"unknown provider {chosen!r}; available: {', '.join(available())}"
        ) from None
    if chosen != "fake":
        kwargs.setdefault("settings", settings)
        kwargs.setdefault("model", model or settings.model)
    return factory(**kwargs)


def from_env(env: dict[str, str] | None = None) -> CompletionProvider:
    """Build the provider the environment asks for."""
    settings = Settings.from_env(env)
    return create(settings.provider, settings=settings)


__all__ = [
    "DEFAULT_MODELS",
    "PROVIDERS",
    "FakeCompletionProvider",
    "RecordedCall",
    "available",
    "create",
    "from_env",
    "register",
]
