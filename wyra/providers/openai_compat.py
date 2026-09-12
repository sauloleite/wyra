"""OpenAI-compatible adapter: OpenAI, Azure (v1 endpoint), vLLM, LM Studio, Groq, Ollama.

One adapter covers every server that speaks the chat-completions API; which one you reach
is a base URL, not a code change. Needs ``pip install 'wyra[openai]'``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..config import DEFAULT_MODELS, Settings
from ..domain import Completion, Message
from ..errors import ConfigError, ProviderError

SCHEMA_NAME = "wyra_examples"


class OpenAICompatibleProvider:
    """Chat completions with JSON-schema structured output."""

    name = "openai"

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
        settings: Settings | None = None,
        strict_schema: bool = True,
    ) -> None:
        settings = settings or Settings.from_env()
        self.model = model or settings.model or DEFAULT_MODELS["openai"]
        self.strict_schema = strict_schema
        base_url = base_url or settings.openai_base_url
        if client is not None:
            self._client = client
            return

        key = api_key or settings.openai_api_key
        if not key and not base_url:
            raise ConfigError(
                "OPENAI_API_KEY is not set. Export it, pass api_key=..., or set "
                "OPENAI_BASE_URL to reach a local server that does not need a key."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ConfigError(
                "provider 'openai' requires the optional dependency: pip install 'wyra[openai]'"
            ) from exc
        self._client = OpenAI(api_key=key or "not-needed", base_url=base_url)

    def complete(
        self,
        messages: Sequence[Message],
        *,
        json_schema: Mapping[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> Completion:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "temperature": temperature,
        }
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": SCHEMA_NAME,
                    "schema": dict(json_schema),
                    "strict": self.strict_schema,
                },
            }
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # SDK exceptions must not leak past the adapter
            raise ProviderError(f"OpenAI-compatible request failed: {exc}") from exc

        choice = _first(response)
        text = getattr(getattr(choice, "message", None), "content", None)
        if not isinstance(text, str):
            raise ProviderError("OpenAI-compatible response carried no text content")
        usage = getattr(response, "usage", None)
        return Completion(
            text=text,
            model=str(getattr(response, "model", self.model)),
            input_tokens=_as_int(getattr(usage, "prompt_tokens", None)),
            output_tokens=_as_int(getattr(usage, "completion_tokens", None)),
            finish_reason=_finish_reason(getattr(choice, "finish_reason", None)),
        )


def _first(response: Any) -> Any:
    choices = getattr(response, "choices", None)
    if not choices:
        raise ProviderError("OpenAI-compatible response carried no choices")
    return choices[0]


def _finish_reason(value: Any) -> str | None:
    return str(value) if isinstance(value, str) else None


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
