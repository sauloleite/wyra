"""An embedded local model: no daemon, no API key, nothing to install outside pip.

Needs ``pip install 'wyra[local]'``, which brings onnxruntime-genai. That package ships
prebuilt wheels for macOS, Linux and Windows, so no compiler is involved.

Model weights are deliberately *not* part of any package. PyPI caps a single file at
100 MiB and usable weights are hundreds of megabytes, so they are downloaded once into a
user cache by ``modelstore``, never bundled into the wheel.

Structured output uses the engine's own constrained decoding (``set_guidance``), which
restricts token selection to what the JSON schema allows. The schema is enforced while the
text is produced, not merely requested in a prompt.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..domain import Completion, Message
from ..errors import ConfigError, ProviderError

GUIDANCE_TYPE = "json_schema"


class EmbeddedProvider:
    """Run a small local model in this process through onnxruntime-genai.

    ``model`` names an entry in the model catalogue; ``model_dir`` points at a directory
    that already holds ``genai_config.json``. Loading is deferred to the first call, and
    the loaded model is reused afterwards, because loading costs seconds.
    """

    name = "local"

    def __init__(
        self,
        model: str | None = None,
        *,
        model_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
        max_new_tokens: int = 1024,
        download: bool = False,
        runtime: Any | None = None,
        settings: Any | None = None,
    ) -> None:
        from .modelstore import DEFAULT_MODEL

        self.model = model or DEFAULT_MODEL
        self.max_new_tokens = max_new_tokens
        self._model_dir = Path(model_dir) if model_dir is not None else None
        self._cache_dir = cache_dir
        self._download = download
        self._runtime = runtime
        self._loaded: tuple[Any, Any] | None = None

    # --- engine plumbing -------------------------------------------------------

    def _engine(self) -> Any:
        if self._runtime is not None:
            return self._runtime
        try:
            import onnxruntime_genai
        except ImportError as exc:
            raise ConfigError(
                "provider 'local' requires the optional dependency: pip install 'wyra[local]'"
            ) from exc
        self._runtime = onnxruntime_genai
        return onnxruntime_genai

    def model_path(self) -> Path:
        """Where the weights live, resolving (and optionally fetching) them on demand."""
        if self._model_dir is None:
            from .modelstore import resolve

            self._model_dir = resolve(
                self.model, cache_dir=self._cache_dir, download=self._download
            )
        return self._model_dir

    def _load(self) -> tuple[Any, Any]:
        if self._loaded is not None:
            return self._loaded
        engine = self._engine()
        path = self.model_path()
        try:
            model = engine.Model(str(path))
            tokenizer = engine.Tokenizer(model)
        except Exception as exc:
            raise ProviderError(f"cannot load the local model at {path}: {exc}") from exc
        self._loaded = (model, tokenizer)
        return self._loaded

    # --- the port --------------------------------------------------------------

    def complete(
        self,
        messages: Sequence[Message],
        *,
        json_schema: Mapping[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> Completion:
        engine = self._engine()
        model, tokenizer = self._load()
        prompt = self._render(tokenizer, messages)

        try:
            tokens = tokenizer.encode(prompt)
            params = engine.GeneratorParams(model)
            params.set_search_options(
                max_length=len(tokens) + self.max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 0.0),
            )
            if json_schema is not None:
                params.set_guidance(GUIDANCE_TYPE, json.dumps(dict(json_schema)))
            generator = engine.Generator(model, params)
            generator.append_tokens(tokens)
            while not generator.is_done():
                generator.generate_next_token()
            sequence = generator.get_sequence(0)
        except Exception as exc:
            raise ProviderError(f"local generation failed: {exc}") from exc

        produced = list(sequence)[len(tokens) :]
        text = tokenizer.decode(produced) if produced else ""
        return Completion(
            text=text,
            model=self.model,
            input_tokens=len(tokens),
            output_tokens=len(produced),
            finish_reason="length" if len(produced) >= self.max_new_tokens else "stop",
        )

    @staticmethod
    def _render(tokenizer: Any, messages: Sequence[Message]) -> str:
        """Use the model's own chat template, falling back to a plain transcript."""
        payload = [{"role": m.role.value, "content": m.content} for m in messages]
        try:
            rendered = tokenizer.apply_chat_template(
                messages=json.dumps(payload), add_generation_prompt=True
            )
        except Exception:
            rendered = None
        if isinstance(rendered, str) and rendered.strip():
            return rendered
        transcript = "\n\n".join(f"{m.role.value}: {m.content}" for m in messages)
        return f"{transcript}\n\nassistant:"
