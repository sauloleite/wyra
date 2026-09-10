"""Google Gemini adapter. Needs ``pip install 'wyra[gemini]'``.

The API key comes from the environment, never from source. The previous version of this
library shipped a key in its published package; that mistake is what this module's
constructor is designed to make impossible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..config import DEFAULT_MODELS, Settings
from ..domain import Completion, Message, Role
from ..errors import ConfigError, ProviderError


class GeminiProvider:
    """Generate content with a JSON mime type and a response schema."""

    name = "gemini"

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        client: Any | None = None,
        settings: Settings | None = None,
    ) -> None:
        settings = settings or Settings.from_env()
        self.model = model or settings.model or DEFAULT_MODELS["gemini"]
        if client is not None:
            self._client = client
            return

        key = api_key or settings.gemini_api_key
        if not key:
            raise ConfigError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set. Export it or pass "
                "api_key=... to GeminiProvider."
            )
        try:
            from google import genai
        except ImportError as exc:
            raise ConfigError(
                "provider 'gemini' requires the optional dependency: pip install 'wyra[gemini]'"
            ) from exc
        self._client = genai.Client(api_key=key)

    def complete(
        self,
        messages: Sequence[Message],
        *,
        json_schema: Mapping[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> Completion:
        system = "\n\n".join(m.content for m in messages if m.role is Role.SYSTEM)
        contents = [
            {
                "role": "model" if m.role is Role.ASSISTANT else "user",
                "parts": [{"text": m.content}],
            }
            for m in messages
            if m.role is not Role.SYSTEM
        ]
        config: dict[str, Any] = {"temperature": temperature}
        if system:
            config["system_instruction"] = system
        if json_schema is not None:
            config["response_mime_type"] = "application/json"
            config["response_json_schema"] = dict(json_schema)

        try:
            response = self._client.models.generate_content(
                model=self.model, contents=contents, config=config
            )
        except Exception as exc:  # SDK exceptions must not leak past the adapter
            raise ProviderError(f"Gemini request failed: {exc}") from exc

        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise ProviderError("Gemini response carried no text content")
        usage = getattr(response, "usage_metadata", None)
        return Completion(
            text=text,
            model=self.model,
            input_tokens=_as_int(getattr(usage, "prompt_token_count", None)),
            output_tokens=_as_int(getattr(usage, "candidates_token_count", None)),
            finish_reason=_finish_reason(response),
        )


def _finish_reason(response: Any) -> str | None:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    reason = getattr(candidates[0], "finish_reason", None)
    if reason is None:
        return None
    name = getattr(reason, "name", None)
    text = str(name or reason).upper()
    return "length" if "MAX_TOKEN" in text else text.lower()


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
