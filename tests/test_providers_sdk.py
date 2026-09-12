"""Adapter mapping against stub clients. No SDK import, no network, no credentials."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from wyra.domain import Message, Role
from wyra.errors import ProviderError
from wyra.providers.gemini import GeminiProvider
from wyra.providers.openai_compat import OpenAICompatibleProvider

MESSAGES = [Message(Role.SYSTEM, "seja fiel"), Message(Role.USER, "gere pares")]
SCHEMA: dict[str, Any] = {"type": "object", "properties": {"pairs": {"type": "array"}}}


class StubOpenAI:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.captured: dict[str, Any] = {}
        self._response = response
        self._error = error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.captured = kwargs
        if self._error:
            raise self._error
        return self._response


def openai_response(content: str | None = '{"pairs": []}') -> Any:
    return SimpleNamespace(
        model="gpt-4o-mini",
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
    )


def test_openai_maps_schema_messages_and_usage() -> None:
    client = StubOpenAI(openai_response())
    provider = OpenAICompatibleProvider("gpt-4o-mini", client=client)
    completion = provider.complete(MESSAGES, json_schema=SCHEMA, temperature=0.4)

    assert provider.name == "openai"
    assert client.captured["model"] == "gpt-4o-mini"
    assert client.captured["temperature"] == 0.4
    assert client.captured["messages"][0] == {"role": "system", "content": "seja fiel"}
    response_format = client.captured["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["schema"] == SCHEMA
    assert response_format["json_schema"]["strict"] is True
    assert completion.text == '{"pairs": []}'
    assert (completion.input_tokens, completion.output_tokens) == (11, 7)
    assert completion.finish_reason == "stop"


def test_openai_omits_the_schema_and_can_relax_strictness() -> None:
    client = StubOpenAI(openai_response())
    OpenAICompatibleProvider(client=client).complete(MESSAGES)
    assert "response_format" not in client.captured

    client = StubOpenAI(openai_response())
    OpenAICompatibleProvider(client=client, strict_schema=False).complete(
        MESSAGES, json_schema=SCHEMA
    )
    assert client.captured["response_format"]["json_schema"]["strict"] is False


def test_openai_failures_never_leak_sdk_exceptions() -> None:
    class SdkError(Exception):
        pass

    provider = OpenAICompatibleProvider(client=StubOpenAI(error=SdkError("429 rate limit")))
    with pytest.raises(ProviderError, match="429 rate limit"):
        provider.complete(MESSAGES)

    with pytest.raises(ProviderError, match="no choices"):
        OpenAICompatibleProvider(client=StubOpenAI(SimpleNamespace(choices=[]))).complete(MESSAGES)
    with pytest.raises(ProviderError, match="no text"):
        OpenAICompatibleProvider(client=StubOpenAI(openai_response(None))).complete(MESSAGES)


class StubGemini:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.captured: dict[str, Any] = {}
        self._response = response
        self._error = error
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, **kwargs: Any) -> Any:
        self.captured = kwargs
        if self._error:
            raise self._error
        return self._response


def gemini_response(text: str | None = '{"pairs": []}', finish: str = "STOP") -> Any:
    return SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(prompt_token_count=9, candidates_token_count=5),
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name=finish))],
    )


def test_gemini_maps_system_instruction_contents_and_schema() -> None:
    client = StubGemini(gemini_response())
    provider = GeminiProvider("gemini-2.5-flash", client=client)
    completion = provider.complete(MESSAGES, json_schema=SCHEMA, temperature=0.1)

    assert provider.name == "gemini"
    config = client.captured["config"]
    assert config["system_instruction"] == "seja fiel"
    assert config["response_mime_type"] == "application/json"
    assert config["response_json_schema"] == SCHEMA
    assert config["temperature"] == 0.1
    assert client.captured["contents"] == [{"role": "user", "parts": [{"text": "gere pares"}]}]
    assert completion.text == '{"pairs": []}'
    assert (completion.input_tokens, completion.output_tokens) == (9, 5)
    assert completion.finish_reason == "stop"


def test_gemini_reports_truncation_and_failures() -> None:
    truncated = GeminiProvider(client=StubGemini(gemini_response(finish="MAX_TOKENS")))
    assert truncated.complete(MESSAGES).finish_reason == "length"

    with pytest.raises(ProviderError, match="no text"):
        GeminiProvider(client=StubGemini(gemini_response(""))).complete(MESSAGES)
    with pytest.raises(ProviderError, match="Gemini request failed"):
        GeminiProvider(client=StubGemini(error=RuntimeError("quota"))).complete(MESSAGES)


def test_gemini_without_candidates_reports_no_finish_reason() -> None:
    response = SimpleNamespace(text="{}", usage_metadata=None, candidates=[])
    assert GeminiProvider(client=StubGemini(response)).complete(MESSAGES).finish_reason is None


def test_a_candidate_without_a_finish_reason_reports_none() -> None:
    response = SimpleNamespace(
        text="{}",
        usage_metadata=None,
        candidates=[SimpleNamespace(finish_reason=None)],
    )
    assert GeminiProvider(client=StubGemini(response)).complete(MESSAGES).finish_reason is None
