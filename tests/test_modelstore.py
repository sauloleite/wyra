"""The embedded-model catalogue and cache. No network: the opener is injected."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from wyra.errors import ConfigError, ProviderError
from wyra.providers import modelstore as ms

ENTRY = ms.CatalogEntry(
    name="test-model",
    repo="acme/test-onnx",
    revision="0123456789abcdef0123456789abcdef01234567",
    prefix="cpu/int4",
    size_mb=12,
    license="mit",
    parameters="0.1B",
    note="only for tests",
)

FILES = {
    "cpu/int4/genai_config.json": b'{"model": {}}',
    "cpu/int4/model.onnx": b"ONNX",
    "cpu/int4/nested/tokenizer.json": b"TOK",
    "cpu/int4/.gitattributes": b"skip me",
    "other/ignored.bin": b"not this one",
}


def api_payload(paths: dict[str, bytes] | None = None) -> dict[str, Any]:
    source = FILES if paths is None else paths
    return {"siblings": [{"rfilename": p, "size": len(b)} for p, b in source.items()]}


def make_opener(
    *, payload: Any = None, files: dict[str, bytes] | None = None, fail: str | None = None
) -> Any:
    body = api_payload() if payload is None else payload
    blobs = FILES if files is None else files

    def opener(request: urllib.request.Request, timeout: float) -> Any:
        url = request.full_url
        if fail is not None and fail in url:
            raise urllib.error.HTTPError(url, 503, "unavailable", {}, None)  # type: ignore[arg-type]
        if "/api/models/" in url:
            raw = body if isinstance(body, bytes) else json.dumps(body).encode()
            return io.BytesIO(raw)
        path = url.split("/resolve/", 1)[1].split("/", 1)[1]
        return io.BytesIO(blobs[path])

    return opener


def test_the_shipped_catalogue_is_coherent() -> None:
    assert ms.DEFAULT_MODEL in ms.CATALOG
    assert ms.CATALOG[ms.DEFAULT_MODEL].license == "mit"
    for entry in ms.CATALOG.values():
        assert len(entry.revision) == 40, entry.name
        assert entry.size_mb > 0 and entry.note
        assert entry.url.startswith("https://huggingface.co/")
    assert ms.CATALOG["qwen2.5-0.5b"].license == ms.UNDECLARED
    assert not ms.CATALOG["qwen2.5-0.5b"].license_declared
    assert ms.CATALOG["phi-3.5-mini"].license_declared
    assert ms.available() == ["phi-3.5-mini", "phi-3-mini", "qwen2.5-0.5b"]


def test_unknown_model_lists_the_options() -> None:
    with pytest.raises(ConfigError, match="unknown local model"):
        ms.get_entry("llama-999b")


def test_cache_root_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WYRA_CACHE_DIR", str(tmp_path / "from-env"))
    assert ms.cache_root() == tmp_path / "from-env"
    assert ms.cache_root(tmp_path / "explicit") == tmp_path / "explicit"
    monkeypatch.delenv("WYRA_CACHE_DIR")
    default = ms.cache_root()
    assert default.is_absolute() and default.name == "models"


def test_paths_accept_a_name_or_an_entry(tmp_path: Path) -> None:
    assert ms.model_dir(ENTRY, tmp_path) == tmp_path / "test-model"
    assert ms.model_dir("phi-3-mini", tmp_path) == tmp_path / "phi-3-mini"
    assert not ms.is_cached(ENTRY, tmp_path)
    target = tmp_path / "phi-3-mini"
    target.mkdir()
    (target / ms.CONFIG_FILE).write_text("{}", encoding="utf-8")
    assert ms.is_cached("phi-3-mini", tmp_path)
    assert ms.cached_models(tmp_path) == ["phi-3-mini"]


def test_resolve_returns_a_cached_model_without_downloading(tmp_path: Path) -> None:
    target = tmp_path / ms.DEFAULT_MODEL
    target.mkdir()
    (target / ms.CONFIG_FILE).write_text("{}", encoding="utf-8")
    assert ms.resolve(cache_dir=tmp_path) == target


def test_resolve_never_downloads_silently(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as info:
        ms.resolve("qwen2.5-0.5b", cache_dir=tmp_path)
    message = str(info.value)
    assert "874 MB" in message
    assert "undeclared" in message
    assert "wyra setup --download qwen2.5-0.5b" in message


def test_list_files_keeps_the_variant_and_drops_the_rest() -> None:
    files = ms.list_files(ENTRY, opener=make_opener())
    assert [path for path, _ in files] == [
        "cpu/int4/genai_config.json",
        "cpu/int4/model.onnx",
        "cpu/int4/nested/tokenizer.json",
    ]


def test_list_files_rejects_a_repository_the_engine_cannot_load() -> None:
    payload = api_payload({"cpu/int4/model.onnx": b"ONNX"})
    with pytest.raises(ProviderError, match="genai_config.json"):
        ms.list_files(ENTRY, opener=make_opener(payload=payload))


def test_list_files_reports_transport_and_shape_problems() -> None:
    with pytest.raises(ProviderError, match="HTTP 503"):
        ms.list_files(ENTRY, opener=make_opener(fail="/api/models/"), attempts=1)
    with pytest.raises(ProviderError, match="invalid JSON"):
        ms.list_files(ENTRY, opener=make_opener(payload=b"<html>"))
    with pytest.raises(ProviderError, match="unexpected"):
        ms.list_files(ENTRY, opener=make_opener(payload={"siblings": "nope"}))


def test_fetch_writes_the_variant_flat_with_lineage(tmp_path: Path) -> None:
    seen: list[tuple[str, int, int]] = []
    target = ms.fetch(
        ENTRY, cache_dir=tmp_path, opener=make_opener(), progress=lambda *args: seen.append(args)
    )

    assert target == tmp_path / "test-model"
    assert (target / "genai_config.json").read_bytes() == b'{"model": {}}'
    assert (target / "model.onnx").read_bytes() == b"ONNX"
    assert (target / "nested" / "tokenizer.json").read_bytes() == b"TOK"
    assert not (target / ".gitattributes").exists()
    assert not (target / "cpu").exists()
    assert [name for name, _, _ in seen] == [
        "genai_config.json",
        "model.onnx",
        "nested/tokenizer.json",
    ]
    assert seen[-1][1:] == (3, 3)

    lineage = json.loads((target / ms.LINEAGE_FILE).read_text(encoding="utf-8"))
    assert lineage["repo"] == "acme/test-onnx"
    assert lineage["revision"] == ENTRY.revision
    assert lineage["license"] == "mit"
    assert lineage["bytes"] == sum(
        len(FILES[p]) for p in FILES if p.startswith("cpu/int4") and "git" not in p
    )
    assert ms.lineage(ENTRY, tmp_path)["revision"] == ENTRY.revision
    assert ms.is_cached(ENTRY, tmp_path)


def test_fetch_leaves_nothing_behind_when_it_fails(tmp_path: Path) -> None:
    with pytest.raises(ProviderError, match="HTTP 503"):
        ms.fetch(ENTRY, cache_dir=tmp_path, opener=make_opener(fail="model.onnx"), attempts=1)
    assert not (tmp_path / "test-model").exists()
    assert not (tmp_path / "test-model.partial").exists()


def test_fetch_replaces_a_previous_download_and_stale_staging(tmp_path: Path) -> None:
    stale = tmp_path / "test-model.partial"
    stale.mkdir(parents=True)
    (stale / "junk").write_text("x", encoding="utf-8")
    target = tmp_path / "test-model"
    target.mkdir()
    (target / "old.bin").write_text("old", encoding="utf-8")

    ms.fetch(ENTRY, cache_dir=tmp_path, opener=make_opener())
    assert not (target / "old.bin").exists()
    assert (target / "model.onnx").exists()
    assert not stale.exists()


def test_lineage_is_empty_when_absent_or_corrupt(tmp_path: Path) -> None:
    assert ms.lineage(ENTRY, tmp_path) == {}
    target = tmp_path / "test-model"
    target.mkdir()
    (target / ms.LINEAGE_FILE).write_text("not json", encoding="utf-8")
    assert ms.lineage(ENTRY, tmp_path) == {}
    (target / ms.LINEAGE_FILE).write_text("[1, 2]", encoding="utf-8")
    assert ms.lineage(ENTRY, tmp_path) == {}


def test_catalog_rows_carry_what_setup_prints(tmp_path: Path) -> None:
    rows = ms.catalog_rows(tmp_path)
    assert [row["name"] for row in rows] == ms.available()
    default = next(row for row in rows if row["default"])
    assert default["name"] == ms.DEFAULT_MODEL
    assert all(row["cached"] is False for row in rows)
    assert all(row["size_mb"] > 0 and row["parameters"] for row in rows)


class FlakyOpener:
    """Fails a set number of times, on the listing or on the file, recording every call.

    The two are separate on purpose: a test about a rate-limited download must not have its
    failures swallowed by the listing that precedes it.
    """

    def __init__(
        self,
        *,
        failures: int,
        scope: str = "blob",
        error: Exception | None = None,
        payload: Any = None,
        blob: bytes = b"WEIGHTS",
        mid_stream: bool = False,
    ) -> None:
        self.failures = failures
        self.scope = scope
        self.error = error
        self.payload = api_payload() if payload is None else payload
        self.blob = blob
        self.mid_stream = mid_stream
        self.attempts = 0
        self.listing_attempts = 0
        self.blob_attempts = 0

    def __call__(self, request: urllib.request.Request, timeout: float) -> Any:
        self.attempts += 1
        listing = "/api/models/" in request.full_url
        if listing:
            self.listing_attempts += 1
            if self.scope == "listing" and self.listing_attempts <= self.failures:
                return self._fail(request)
            return io.BytesIO(json.dumps(self.payload).encode())
        self.blob_attempts += 1
        if self.scope == "blob" and self.blob_attempts <= self.failures:
            return self._fail(request)
        return io.BytesIO(self.blob)

    def _fail(self, request: urllib.request.Request) -> Any:
        if self.mid_stream:
            return HalfBrokenResponse(self.blob[:3])
        raise self.error or urllib.error.HTTPError(
            request.full_url, 429, "Too Many Requests", {}, None
        )  # type: ignore[arg-type]


class HalfBrokenResponse(io.BytesIO):
    """Hands over a few bytes, then drops the connection, like a real interrupted download."""

    def read(self, size: int = -1) -> bytes:
        chunk = super().read(size)
        if chunk:
            return chunk
        raise urllib.error.URLError("connection reset")


class RecordingSleeper:
    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def one_file_entry() -> ms.CatalogEntry:
    return ms.CatalogEntry(
        name="retry-model",
        repo="acme/retry-onnx",
        revision="f" * 40,
        prefix="",
        size_mb=1,
        license="mit",
        parameters="0.1B",
        note="retry fixture",
    )


def single_file_payload() -> dict[str, Any]:
    return api_payload({ms.CONFIG_FILE: b"WEIGHTS"})


def test_a_rate_limited_download_is_retried_with_growing_delays(tmp_path: Path) -> None:
    opener = FlakyOpener(failures=2, payload=single_file_payload())
    sleeper = RecordingSleeper()
    target = ms.fetch(
        one_file_entry(), cache_dir=tmp_path, opener=opener, sleeper=sleeper, attempts=5
    )

    assert (target / ms.CONFIG_FILE).read_bytes() == b"WEIGHTS"
    assert sleeper.delays == [2.0, 4.0]
    assert opener.blob_attempts == 3
    assert opener.listing_attempts == 1


def test_a_numeric_retry_after_is_honoured(tmp_path: Path) -> None:
    def opener(request: urllib.request.Request, timeout: float) -> Any:
        opener.calls = getattr(opener, "calls", 0) + 1  # type: ignore[attr-defined]
        if opener.calls == 1:  # type: ignore[attr-defined]
            raise urllib.error.HTTPError(
                request.full_url, 429, "slow down", {"Retry-After": "7"}, None
            )  # type: ignore[arg-type]
        return io.BytesIO(json.dumps(single_file_payload()).encode())

    sleeper = RecordingSleeper()
    assert ms.list_files(one_file_entry(), opener=opener, sleeper=sleeper)
    assert sleeper.delays == [7.0]


def test_an_absurd_retry_after_is_capped(tmp_path: Path) -> None:
    def opener(request: urllib.request.Request, timeout: float) -> Any:
        raise urllib.error.HTTPError(
            request.full_url, 429, "slow down", {"Retry-After": "99999"}, None
        )  # type: ignore[arg-type]

    sleeper = RecordingSleeper()
    with pytest.raises(ProviderError, match="HTTP 429"):
        ms.list_files(one_file_entry(), opener=opener, sleeper=sleeper, attempts=2)
    assert sleeper.delays == [ms.MAX_BACKOFF_SECONDS]


def test_a_nonsense_retry_after_falls_back_to_backoff() -> None:
    def opener(request: urllib.request.Request, timeout: float) -> Any:
        raise urllib.error.HTTPError(
            request.full_url, 503, "later", {"Retry-After": "tomorrow"}, None
        )  # type: ignore[arg-type]

    sleeper = RecordingSleeper()
    with pytest.raises(ProviderError, match="HTTP 503"):
        ms.list_files(one_file_entry(), opener=opener, sleeper=sleeper, attempts=2)
    assert sleeper.delays == [2.0]


def test_a_persistent_rate_limit_eventually_gives_up(tmp_path: Path) -> None:
    opener = FlakyOpener(failures=99, payload=single_file_payload())
    sleeper = RecordingSleeper()
    with pytest.raises(ProviderError, match="HTTP 429"):
        ms.fetch(one_file_entry(), cache_dir=tmp_path, opener=opener, sleeper=sleeper, attempts=3)
    assert opener.blob_attempts == 3
    assert len(sleeper.delays) == 2
    assert not (tmp_path / "retry-model").exists()


def test_a_client_error_is_not_retried() -> None:
    error = urllib.error.HTTPError("u", 404, "gone", {}, None)  # type: ignore[arg-type]
    opener = FlakyOpener(failures=99, scope="listing", error=error)
    sleeper = RecordingSleeper()
    with pytest.raises(ProviderError, match="HTTP 404"):
        ms.list_files(one_file_entry(), opener=opener, sleeper=sleeper)
    assert opener.listing_attempts == 1
    assert sleeper.delays == []


def test_a_transport_failure_is_retried() -> None:
    opener = FlakyOpener(
        failures=1,
        scope="listing",
        error=urllib.error.URLError("reset"),
        payload=single_file_payload(),
    )
    sleeper = RecordingSleeper()
    assert ms.list_files(one_file_entry(), opener=opener, sleeper=sleeper)
    assert sleeper.delays == [2.0]
    assert opener.listing_attempts == 2

    always = FlakyOpener(failures=99, scope="listing", error=urllib.error.URLError("reset"))
    with pytest.raises(ProviderError, match="cannot reach"):
        ms.list_files(one_file_entry(), opener=always, sleeper=RecordingSleeper(), attempts=2)


def test_an_interrupted_download_restarts_instead_of_appending(tmp_path: Path) -> None:
    opener = FlakyOpener(
        failures=1, payload=single_file_payload(), blob=b"WEIGHTS", mid_stream=True
    )
    sleeper = RecordingSleeper()
    target = ms.fetch(
        one_file_entry(), cache_dir=tmp_path, opener=opener, sleeper=sleeper, attempts=3
    )
    # the first attempt delivered "WEI" before dropping; the file must hold the whole body once
    assert (target / ms.CONFIG_FILE).read_bytes() == b"WEIGHTS"
    assert sleeper.delays == [2.0]


def test_backoff_grows_then_stops_growing() -> None:
    delays = [ms._backoff(attempt) for attempt in range(1, 9)]
    assert delays[:3] == [2.0, 4.0, 8.0]
    assert delays[-1] == ms.MAX_BACKOFF_SECONDS
    assert all(later >= earlier for earlier, later in zip(delays, delays[1:], strict=False))


def test_read_lineage_reads_a_directory_directly(tmp_path: Path) -> None:
    assert ms.read_lineage(tmp_path) == {}
    (tmp_path / ms.LINEAGE_FILE).write_text('{"revision": "abc"}', encoding="utf-8")
    assert ms.read_lineage(tmp_path) == {"revision": "abc"}
    assert ms.lineage("phi-3-mini", tmp_path.parent) == {}
    target = tmp_path.parent / "phi-3-mini"
    target.mkdir(exist_ok=True)
    (target / ms.LINEAGE_FILE).write_text('{"revision": "def"}', encoding="utf-8")
    assert ms.lineage("phi-3-mini", tmp_path.parent) == {"revision": "def"}
