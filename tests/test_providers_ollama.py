from __future__ import annotations

import json
from typing import Any
from urllib import error, request

import pytest

from wyra.domain import Message, Role
from wyra.errors import ProviderError
from wyra.providers.ollama import OllamaProvider

MESSAGES = [Message(Role.SYSTEM, "seja fiel"), Message(Role.USER, "gere pares")]
SCHEMA = {"type": "object", "properties": {"pairs": {"type": "array"}}}


class FakeResponse:
    def __init__(self, payload: Any) -> None:
        self._raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def opener_for(payload: Any, sink: list[request.Request] | None = None):
    def opener(req: request.Request, timeout: float) -> FakeResponse:
        if sink is not None:
            sink.append(req)
        return FakeResponse(payload)

    return opener


def test_request_shape_and_completion_mapping() -> None:
    sent: list[request.Request] = []
    provider = OllamaProvider(
        "llama3.2:1b",
        host="box:11434",
        num_predict=256,
        opener=opener_for(
            {
                "model": "llama3.2:1b",
                "message": {"role": "assistant", "content": '{"pairs": []}'},
                "prompt_eval_count": 120,
                "eval_count": 42,
                "done_reason": "stop",
            },
            sent,
        ),
    )
    completion = provider.complete(MESSAGES, json_schema=SCHEMA, temperature=0.3)

    assert provider.name == "ollama" and provider.host == "http://box:11434"
    assert sent[0].full_url == "http://box:11434/api/chat"
    assert sent[0].get_method() == "POST"
    body = json.loads(sent[0].data or b"{}")
    assert body["model"] == "llama3.2:1b"
    assert body["stream"] is False
    assert body["format"] == SCHEMA
    assert body["options"] == {"temperature": 0.3, "num_predict": 256}
    assert body["messages"] == [
        {"role": "system", "content": "seja fiel"},
        {"role": "user", "content": "gere pares"},
    ]
    assert completion.text == '{"pairs": []}'
    assert (completion.input_tokens, completion.output_tokens) == (120, 42)
    assert completion.finish_reason == "stop"


def test_schema_is_omitted_when_not_requested() -> None:
    sent: list[request.Request] = []
    provider = OllamaProvider(opener=opener_for({"message": {"content": "hi"}}, sent))
    provider.complete(MESSAGES)
    assert "format" not in json.loads(sent[0].data or b"{}")


def test_truncation_is_reported_through_finish_reason() -> None:
    provider = OllamaProvider(
        opener=opener_for({"message": {"content": "{"}, "done_reason": "length"})
    )
    assert provider.complete(MESSAGES).finish_reason == "length"


def test_http_error_surfaces_the_server_message() -> None:
    message = json.dumps({"error": "model requires more system memory (3.5 GiB)"}).encode()

    def opener(req: request.Request, timeout: float) -> FakeResponse:
        raise error.HTTPError(req.full_url, 500, "Server Error", {}, None)  # type: ignore[arg-type]

    def opener_with_body(req: request.Request, timeout: float) -> FakeResponse:
        raise error.HTTPError(
            req.full_url, 500, "Server Error", {}, __import__("io").BytesIO(message)
        )

    with pytest.raises(ProviderError, match="HTTP 500"):
        OllamaProvider(opener=opener).complete(MESSAGES)
    with pytest.raises(ProviderError, match="more system memory"):
        OllamaProvider(opener=opener_with_body).complete(MESSAGES)


def test_connection_failure_names_the_host_and_the_fix() -> None:
    def opener(req: request.Request, timeout: float) -> FakeResponse:
        raise error.URLError("Connection refused")

    with pytest.raises(ProviderError) as info:
        OllamaProvider(host="localhost:9999", opener=opener).complete(MESSAGES)
    assert "localhost:9999" in str(info.value)
    assert "ollama serve" in str(info.value)


def test_timeouts_and_other_os_errors_are_wrapped() -> None:
    def opener(req: request.Request, timeout: float) -> FakeResponse:
        raise TimeoutError("timed out")

    with pytest.raises(ProviderError, match="cannot reach Ollama"):
        OllamaProvider(opener=opener).complete(MESSAGES)


def test_malformed_and_error_payloads() -> None:
    with pytest.raises(ProviderError, match="invalid JSON"):
        OllamaProvider(opener=opener_for(b"not json")).complete(MESSAGES)
    with pytest.raises(ProviderError, match="Ollama error"):
        OllamaProvider(opener=opener_for({"error": "model not found"})).complete(MESSAGES)
    with pytest.raises(ProviderError, match="unexpected"):
        OllamaProvider(opener=opener_for({"message": {"role": "assistant"}})).complete(MESSAGES)
    with pytest.raises(ProviderError, match="unexpected"):
        OllamaProvider(opener=opener_for([1, 2])).complete(MESSAGES)


def test_settings_supply_the_host_and_model() -> None:
    from wyra.config import Settings

    settings = Settings.from_env({"OLLAMA_HOST": "gpu-box:11434", "WYRA_MODEL": "qwen3:4b"})
    provider = OllamaProvider(settings=settings, opener=opener_for({"message": {"content": "x"}}))
    assert provider.host == "http://gpu-box:11434"
    assert provider.model == "qwen3:4b"
