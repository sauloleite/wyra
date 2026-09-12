"""Catalogue and cache for embedded models. Weights live outside the package, by design.

PyPI caps a single file at 100 MiB and a usable ONNX model runs from hundreds of megabytes
to a few gigabytes, so no wheel can carry weights. They are fetched once, only when asked
for explicitly, into a user cache, the way spaCy, NLTK and Hugging Face all do it.

Downloading uses the standard library, so the ``local`` extra stays a single package. Every
entry pins a repository revision, so the same name always fetches the same bytes and the
dataset manifest can say which weights produced it.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..errors import ConfigError, ProviderError

HF_API = "https://huggingface.co/api/models"
HF_FILES = "https://huggingface.co"
CONFIG_FILE = "genai_config.json"
LINEAGE_FILE = "wyra_model.json"
UNDECLARED = "undeclared"
SKIP_FILES = (".gitattributes",)

# Hugging Face rate-limits large downloads, so a multi-gigabyte fetch has to wait and
# retry rather than abandon the whole model.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
MAX_ATTEMPTS = 5
BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 60.0

Opener = Callable[[urllib.request.Request, float], Any]
Progress = Callable[[str, int, int], None]
Sleeper = Callable[[float], None]


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """One downloadable model. ``prefix`` selects a variant folder inside the repository."""

    name: str
    repo: str
    revision: str
    prefix: str
    size_mb: int
    license: str
    parameters: str
    note: str = ""

    @property
    def license_declared(self) -> bool:
        return self.license != UNDECLARED

    @property
    def url(self) -> str:
        return f"{HF_FILES}/{self.repo}/tree/{self.revision}/{self.prefix}".rstrip("/")


# Only models that ship genai_config.json can be loaded by onnxruntime-genai, which rules
# out most small ONNX exports. Sizes and revisions were measured on 2026-09-12.
CATALOG: dict[str, CatalogEntry] = {
    "phi-3.5-mini": CatalogEntry(
        name="phi-3.5-mini",
        repo="microsoft/Phi-3.5-mini-instruct-onnx",
        revision="7230dcd6c1dd28aab70f263ecc8734ec9d9bcb70",
        prefix="cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4",
        size_mb=2782,
        license="mit",
        parameters="3.8B",
        note="Best answers of the three, and the licence is unambiguous.",
    ),
    "phi-3-mini": CatalogEntry(
        name="phi-3-mini",
        repo="microsoft/Phi-3-mini-4k-instruct-onnx",
        revision="5f5f794c1c23c9d5ee142af85df02a6cc52d6945",
        prefix="cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4",
        size_mb=2726,
        license="mit",
        parameters="3.8B",
        note="Older sibling of phi-3.5-mini; same size, shorter context.",
    ),
    "qwen2.5-0.5b": CatalogEntry(
        name="qwen2.5-0.5b",
        repo="hazemmabbas/Qwen2.5-0.5B-int4-block-32-acc-3-Instruct-onnx-cpu",
        revision="5c2e56e94edeb740724f082a0b2dc433ae095481",
        prefix="",
        size_mb=874,
        license=UNDECLARED,
        parameters="0.5B",
        note=(
            "Smallest download by far, but the repository declares no licence "
            "and it writes weak pairs."
        ),
    ),
}

DEFAULT_MODEL = "phi-3.5-mini"


def available() -> list[str]:
    return list(CATALOG)


def get_entry(name: str) -> CatalogEntry:
    try:
        return CATALOG[name]
    except KeyError:
        raise ConfigError(
            f"unknown local model {name!r}; available: {', '.join(available())}"
        ) from None


def cache_root(cache_dir: str | Path | None = None) -> Path:
    """Where weights are kept. ``WYRA_CACHE_DIR`` overrides the platform default."""
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    override = os.environ.get("WYRA_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or "~/AppData/Local"
        return Path(base).expanduser() / "wyra" / "models"
    if sys.platform == "darwin":
        return Path("~/Library/Caches/wyra/models").expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or "~/.cache"
    return Path(base).expanduser() / "wyra" / "models"


def model_dir(model: str | CatalogEntry, cache_dir: str | Path | None = None) -> Path:
    """Where one model's weights live. Accepts a catalogue name or an entry."""
    entry = model if isinstance(model, CatalogEntry) else get_entry(model)
    return cache_root(cache_dir) / entry.name


def is_cached(model: str | CatalogEntry, cache_dir: str | Path | None = None) -> bool:
    return (model_dir(model, cache_dir) / CONFIG_FILE).is_file()


def cached_models(cache_dir: str | Path | None = None) -> list[str]:
    return [name for name in CATALOG if is_cached(name, cache_dir)]


def resolve(
    name: str | None = None,
    *,
    cache_dir: str | Path | None = None,
    download: bool = False,
    opener: Opener | None = None,
    progress: Progress | None = None,
) -> Path:
    """Return the directory holding the weights, fetching them only if allowed to."""
    entry = get_entry(name or DEFAULT_MODEL)
    target = model_dir(entry, cache_dir)
    if (target / CONFIG_FILE).is_file():
        return target
    if not download:
        raise ConfigError(
            f"local model {entry.name!r} is not downloaded yet: {entry.size_mb} MB, "
            f"licence {entry.license}. Run 'wyra setup --download {entry.name}', "
            "or pass download=True to accept the download from code."
        )
    return fetch(entry, cache_dir=cache_dir, opener=opener, progress=progress)


def list_files(
    entry: CatalogEntry,
    *,
    opener: Opener | None = None,
    attempts: int = MAX_ATTEMPTS,
    sleeper: Sleeper = time.sleep,
) -> list[tuple[str, int]]:
    """The files that make up one variant, as ``(path in repo, size)`` pairs."""
    body = _get_json(
        f"{HF_API}/{entry.repo}?blobs=true&revision={entry.revision}",
        opener,
        attempts=attempts,
        sleeper=sleeper,
    )
    siblings = body.get("siblings") if isinstance(body, dict) else None
    if not isinstance(siblings, list):
        raise ProviderError(f"unexpected Hugging Face response for {entry.repo}")
    files = [
        (item["rfilename"], int(item.get("size") or 0))
        for item in siblings
        if isinstance(item, dict)
        and isinstance(item.get("rfilename"), str)
        and item["rfilename"].startswith(entry.prefix)
        and not item["rfilename"].endswith(SKIP_FILES)
    ]
    if not any(path.endswith(CONFIG_FILE) for path, _ in files):
        raise ProviderError(
            f"{entry.repo} has no {CONFIG_FILE} under {entry.prefix or 'its root'}, "
            "so onnxruntime-genai cannot load it"
        )
    return sorted(files)


def fetch(
    entry: CatalogEntry,
    *,
    cache_dir: str | Path | None = None,
    opener: Opener | None = None,
    progress: Progress | None = None,
    timeout: float = 600.0,
    attempts: int = MAX_ATTEMPTS,
    sleeper: Sleeper = time.sleep,
) -> Path:
    """Download one variant into the cache. Atomic: a partial download never looks done."""
    target = model_dir(entry, cache_dir)
    staging = target.with_name(f"{target.name}.partial")
    files = list_files(entry, opener=opener, attempts=attempts, sleeper=sleeper)

    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        for index, (path, _size) in enumerate(files, start=1):
            relative = path[len(entry.prefix) :].lstrip("/") if entry.prefix else path
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if progress is not None:
                progress(relative, index, len(files))
            _download(
                f"{HF_FILES}/{entry.repo}/resolve/{entry.revision}/{path}",
                destination,
                opener,
                timeout,
                attempts=attempts,
                sleeper=sleeper,
            )
        _write_lineage(staging, entry, files)
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def lineage(name: str, cache_dir: str | Path | None = None) -> dict[str, Any]:
    """What was downloaded for this model, for the dataset manifest."""
    path = model_dir(name, cache_dir) / LINEAGE_FILE
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_lineage(directory: Path, entry: CatalogEntry, files: Sequence[tuple[str, int]]) -> None:
    payload = {
        "name": entry.name,
        "repo": entry.repo,
        "revision": entry.revision,
        "prefix": entry.prefix,
        "license": entry.license,
        "parameters": entry.parameters,
        "bytes": sum(size for _, size in files),
        "files": [path for path, _ in files],
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (directory / LINEAGE_FILE).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _default_opener(request: urllib.request.Request, timeout: float) -> Any:
    return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310 - fixed host


def _get_json(
    url: str,
    opener: Opener | None,
    timeout: float = 30.0,
    *,
    attempts: int = MAX_ATTEMPTS,
    sleeper: Sleeper = time.sleep,
) -> Any:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    raw = _read(request, opener, timeout, attempts=attempts, sleeper=sleeper)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"Hugging Face sent invalid JSON: {raw[:200]!r}") from exc


def _read(
    request: urllib.request.Request,
    opener: Opener | None,
    timeout: float,
    *,
    attempts: int = MAX_ATTEMPTS,
    sleeper: Sleeper = time.sleep,
) -> bytes:
    open_url = opener or _default_opener
    for attempt in range(1, max(attempts, 1) + 1):
        try:
            with open_url(request, timeout) as response:
                data: bytes = response.read()
                return data
        except urllib.error.HTTPError as exc:
            _handle_http(exc, request.full_url, attempt, attempts, sleeper)
        except (urllib.error.URLError, OSError) as exc:
            _handle_transport(exc, request.full_url, attempt, attempts, sleeper)
    raise ProviderError(f"gave up reading {request.full_url}")  # pragma: no cover


def _download(
    url: str,
    destination: Path,
    opener: Opener | None,
    timeout: float,
    *,
    attempts: int = MAX_ATTEMPTS,
    sleeper: Sleeper = time.sleep,
) -> None:
    """Fetch one file, restarting it from scratch on a retry so bytes never duplicate."""
    open_url = opener or _default_opener
    request = urllib.request.Request(url)
    for attempt in range(1, max(attempts, 1) + 1):
        try:
            with open_url(request, timeout) as response, destination.open("wb") as handle:
                shutil.copyfileobj(response, handle, 1 << 20)
            return
        except urllib.error.HTTPError as exc:
            destination.unlink(missing_ok=True)
            _handle_http(exc, url, attempt, attempts, sleeper)
        except (urllib.error.URLError, OSError) as exc:
            destination.unlink(missing_ok=True)
            _handle_transport(exc, url, attempt, attempts, sleeper)
    raise ProviderError(f"gave up downloading {url}")  # pragma: no cover


def _handle_http(
    exc: urllib.error.HTTPError, url: str, attempt: int, attempts: int, sleeper: Sleeper
) -> None:
    """Sleep and let the caller retry, or raise when the status or the budget says stop."""
    if exc.code not in RETRYABLE_STATUS or attempt >= attempts:
        raise ProviderError(f"{url} returned HTTP {exc.code}") from exc
    sleeper(_retry_after(exc) or _backoff(attempt))


def _handle_transport(
    exc: Exception, url: str, attempt: int, attempts: int, sleeper: Sleeper
) -> None:
    if attempt >= attempts:
        raise ProviderError(f"cannot reach {url}: {exc}") from exc
    sleeper(_backoff(attempt))


def _backoff(attempt: int) -> float:
    delay: float = BACKOFF_SECONDS * float(2 ** (attempt - 1))
    return delay if delay < MAX_BACKOFF_SECONDS else MAX_BACKOFF_SECONDS


def _retry_after(exc: urllib.error.HTTPError) -> float | None:
    """Honour a numeric Retry-After, which is what Hugging Face sends with 429."""
    headers: Any = getattr(exc, "headers", None)
    raw: Any = headers.get("Retry-After") if headers is not None else None
    if not raw:
        return None
    try:
        seconds: float = float(raw)
    except (TypeError, ValueError):
        return None
    return seconds if seconds < MAX_BACKOFF_SECONDS else MAX_BACKOFF_SECONDS


def catalog_rows(cache_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Catalogue as plain rows, for ``wyra setup`` to print."""
    return [
        {
            "name": entry.name,
            "parameters": entry.parameters,
            "size_mb": entry.size_mb,
            "license": entry.license,
            "cached": is_cached(entry.name, cache_dir),
            "default": entry.name == DEFAULT_MODEL,
            "note": entry.note,
        }
        for entry in CATALOG.values()
    ]
