"""Ollama adapter: a local model, no API key, no extra dependency.

Uses ``urllib`` from the standard library on purpose, so the free path costs nothing to
install. Structured output goes through Ollama's ``format`` field, which constrains
decoding to a JSON schema.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib import error, request

from ..config import DEFAULT_MODELS, DEFAULT_OLLAMA_HOST, Settings, normalize_host
from ..domain import Completion, Message
from ..errors import ProviderError

Opener = Callable[[request.Request, float], Any]


def _urlopen(req: request.Request, timeout: float) -> Any:
    return request.urlopen(req, timeout=timeout)  # noqa: S310 - the host is caller-controlled


class OllamaProvider:
    """Talk to ``POST /api/chat`` on a local Ollama server."""

    name = "ollama"

    def __init__(
        self,
        model: str | None = None,
        *,
        host: str = DEFAULT_OLLAMA_HOST,
        timeout: float = 120.0,
        num_predict: int = 2048,
        opener: Opener | None = None,
        settings: Settings | None = None,
    ) -> None:
        settings = settings or Settings()
        self.model = model or settings.model or DEFAULT_MODELS["ollama"]
        self.host = normalize_host(host if host != DEFAULT_OLLAMA_HOST else settings.ollama_host)
        self.timeout = timeout
        self.num_predict = num_predict
        self._open = opener or _urlopen

    def complete(
        self,
        messages: Sequence[Message],
        *,
        json_schema: Mapping[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": self.num_predict},
        }
        if json_schema is not None:
            payload["format"] = dict(json_schema)

        body = self._post("/api/chat", payload)
        message = body.get("message")
        if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
            raise ProviderError(f"unexpected Ollama response: {str(body)[:200]}")
        return Completion(
            text=message["content"],
            model=str(body.get("model", self.model)),
            input_tokens=_as_int(body.get("prompt_eval_count")),
            output_tokens=_as_int(body.get("eval_count")),
            finish_reason=_finish_reason(body.get("done_reason")),
        )

    def _post(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        req = request.Request(
            f"{self.host}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._open(req, self.timeout) as response:
                raw = response.read()
        except error.HTTPError as exc:
            raise ProviderError(f"Ollama returned HTTP {exc.code}: {_error_body(exc)}") from exc
        except error.URLError as exc:
            raise ProviderError(
                f"cannot reach Ollama at {self.host}: {exc.reason}. "
                "Is it running? Start it with 'ollama serve'."
            ) from exc
        except OSError as exc:
            raise ProviderError(f"cannot reach Ollama at {self.host}: {exc}") from exc

        try:
            body = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"Ollama sent invalid JSON: {raw[:200]!r}") from exc
        if not isinstance(body, Mapping):
            raise ProviderError(f"unexpected Ollama response: {str(body)[:200]}")
        if "error" in body:
            raise ProviderError(f"Ollama error: {body['error']}")
        return body


def probe(
    host: str = DEFAULT_OLLAMA_HOST, *, timeout: float = 2.0, opener: Opener | None = None
) -> dict[str, Any] | None:
    """Ask a local Ollama for its version and models. ``None`` when it is not reachable.

    Used by ``wyra setup`` to report what is available instead of failing at generation time.
    """
    base = normalize_host(host)
    open_url = opener or _urlopen
    result: dict[str, Any] = {"host": base}
    for path, key in (("/api/version", "version"), ("/api/tags", "models")):
        try:
            with open_url(request.Request(f"{base}{path}"), timeout) as response:
                body = json.loads(response.read())
        except Exception:
            if key == "version":
                return None
            body = {}
        if key == "version":
            result["version"] = (
                str(body.get("version", "unknown")) if isinstance(body, Mapping) else "unknown"
            )
        elif isinstance(body, Mapping) and isinstance(body.get("models"), list):
            result["models"] = [
                str(m.get("name"))
                for m in body["models"]
                if isinstance(m, Mapping) and m.get("name")
            ]
    result.setdefault("models", [])
    return result


def _error_body(exc: error.HTTPError) -> str:
    try:
        raw = exc.read().decode("utf-8", "replace")
    except Exception:  # pragma: no cover - defensive
        return exc.reason if isinstance(exc.reason, str) else str(exc.reason)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw[:300]
    if isinstance(parsed, Mapping) and "error" in parsed:
        return str(parsed["error"])
    return raw[:300]


def _finish_reason(done_reason: Any) -> str | None:
    return str(done_reason) if isinstance(done_reason, str) else None


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
