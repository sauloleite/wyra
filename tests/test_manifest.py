from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from wyra import __version__
from wyra.manifest import Manifest, sha256_of_file, utc_now


def test_utc_now_is_iso_zulu() -> None:
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", utc_now())


def test_sha256_of_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"x" * 3_000_000)
    assert sha256_of_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_round_trip(tmp_path: Path) -> None:
    manifest = Manifest(
        wyra_version=__version__,
        generator={"name": "markdown-sections", "params": {"min_body_chars": 80}},
        sources=[{"source": "a.md", "sha256": "abc", "chars": 10}],
        counts={"generated": 5, "kept": 4},
        split={"validation_fraction": 0.1, "seed": 42},
    )
    path = manifest.write(tmp_path / "nested" / "manifest.json")
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["wyra_version"] == __version__
    assert data["generator"]["params"]["min_body_chars"] == 80
    assert list(data) == [
        "wyra_version",
        "created_at",
        "generator",
        "chunker",
        "sources",
        "curation",
        "counts",
        "errors",
        "split",
        "output",
        "stats",
    ]
    again = Manifest.read(path)
    assert again.as_dict() == manifest.as_dict()


def test_from_dict_ignores_unknown_keys() -> None:
    manifest = Manifest.from_dict({"wyra_version": "9.9.9", "unexpected": True})
    assert manifest.wyra_version == "9.9.9"
    assert manifest.counts == {}
