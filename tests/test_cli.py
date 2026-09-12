from __future__ import annotations

import json
from pathlib import Path

import pytest

from wyra.cli import main


def test_build_markdown_reports_and_writes(fixtures: Path, tmp_path: Path, capsys) -> None:
    code = main(
        [
            "build",
            str(fixtures / "clean_code.md"),
            "-o",
            str(tmp_path),
            "--lang",
            "pt-br",
            "--valid",
            "0.2",
            "--system",
            "Seja um tutor.",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "generated" in out and "train" in out
    assert "normalize" in out
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "validation.jsonl").exists()
    assert (tmp_path / "manifest.json").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generator"]["params"]["lang"] == "pt-br"


def test_build_with_templates_and_a_chosen_format(fixtures: Path, tmp_path: Path) -> None:
    code = main(
        [
            "build",
            str(fixtures / "qa.csv"),
            "-o",
            str(tmp_path),
            "--generator",
            "template",
            "--user",
            "{pergunta}",
            "--assistant",
            "{resposta}",
            "--format",
            "sharegpt",
            "--no-dedup",
            "--lineage",
        ]
    )
    assert code == 0
    first = json.loads((tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["conversations"][0]["from"] == "human"
    assert (tmp_path / "lineage.jsonl").exists()


def test_build_reports_an_empty_dataset_as_failure(tmp_path: Path, capsys) -> None:
    source = tmp_path / "empty.md"
    source.write_text("sem cabeçalhos, só prosa curta", encoding="utf-8")
    assert main(["build", str(source), "-o", str(tmp_path / "out")]) == 1


def test_validate_returns_zero_only_for_a_clean_file(fixtures: Path, capsys) -> None:
    assert main(["validate", str(fixtures / "good.jsonl")]) == 0
    assert "3/3 valid" in capsys.readouterr().out
    assert main(["validate", str(fixtures / "bad.jsonl")]) == 1
    assert "invalid_json" in capsys.readouterr().out
    assert main(["validate", str(fixtures / "good.jsonl"), "--budget", "1"]) == 0
    assert "over budget (1): 3" in capsys.readouterr().out


def test_convert_writes_the_target_format(fixtures: Path, tmp_path: Path, capsys) -> None:
    destination = tmp_path / "out.jsonl"
    code = main(
        ["convert", str(fixtures / "alpaca.jsonl"), "-o", str(destination), "--to", "openai-chat"]
    )
    assert code == 0
    assert "wrote 2 record(s)" in capsys.readouterr().out
    assert json.loads(destination.read_text(encoding="utf-8").splitlines()[0])["messages"]


def test_errors_are_reported_without_a_traceback(tmp_path: Path, capsys) -> None:
    note = tmp_path / "note.txt"
    note.write_text("prosa", encoding="utf-8")
    assert main(["build", str(note), "-o", str(tmp_path / "out")]) == 2
    assert "cannot guess" in capsys.readouterr().err

    assert main(["build", str(tmp_path / "missing.md"), "-o", str(tmp_path / "out")]) == 2
    assert "not found" in capsys.readouterr().err

    assert (
        main(
            [
                "build",
                str(note),
                "-o",
                str(tmp_path / "out"),
                "--generator",
                "llm-qa",
                "--provider",
                "nope",
            ]
        )
        == 2
    )
    assert "unknown provider" in capsys.readouterr().err


def test_version_and_missing_subcommand(capsys) -> None:
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0
    assert "wyra 0.1.0" in capsys.readouterr().out
    with pytest.raises(SystemExit) as info:
        main([])
    assert info.value.code == 2


def test_verbose_logging_is_accepted(fixtures: Path, tmp_path: Path) -> None:
    assert main(["-v", "build", str(fixtures / "qa.csv"), "-o", str(tmp_path)]) == 0


def test_setup_reports_what_is_available(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-value")
    assert main(["setup", "--cache-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out

    assert "core: installed, no dependencies" in out
    assert "markdown" in out and "qa" in out and "template" in out
    assert "llm-qa" not in out.split("generators that need no model:")[1].split("\n")[0]
    for extra in ("tokens", "openai", "gemini", "local"):
        assert extra in out
    assert "ollama" in out
    assert "gemini   credentials set" in out
    assert "super-secret-value" not in out  # a report never prints credential values
    assert "phi-3.5-mini" in out and "2782 MB" in out and "default" in out
    assert "not downloaded" in out
    assert "NO LICENCE" in out  # the undeclared entry is called out
    assert "wyra setup --download" in out


def test_setup_marks_a_cached_model_as_cached(tmp_path: Path, capsys) -> None:
    from wyra.providers import modelstore

    target = tmp_path / "phi-3-mini"
    target.mkdir()
    (target / modelstore.CONFIG_FILE).write_text("{}", encoding="utf-8")
    main(["setup", "--cache-dir", str(tmp_path)])
    out = capsys.readouterr().out
    cached_row = next(line for line in out.splitlines() if "phi-3-mini " in line)
    assert "cached" in cached_row and "not downloaded" not in cached_row


def test_setup_reports_a_missing_credential(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    main(["setup", "--cache-dir", str(tmp_path)])
    assert "gemini   credentials not set" in capsys.readouterr().out


def test_setup_refuses_to_download_without_confirmation(tmp_path: Path, capsys) -> None:
    code = main(["setup", "--download", "qwen2.5-0.5b", "--cache-dir", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 1
    assert "874 MB" in captured.out
    assert "declares no licence" in captured.out
    assert "--yes to accept" in captured.err


def test_setup_downloads_when_accepted(tmp_path: Path, monkeypatch, capsys) -> None:
    from wyra.providers import modelstore

    calls: list[str] = []

    def fake_fetch(entry, *, cache_dir=None, progress=None, **kwargs):
        calls.append(entry.name)
        if progress is not None:
            progress("genai_config.json", 1, 1)
        target = modelstore.model_dir(entry, cache_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / modelstore.CONFIG_FILE).write_text("{}", encoding="utf-8")
        return target

    monkeypatch.setattr(modelstore, "fetch", fake_fetch)
    code = main(["setup", "--download", "phi-3.5-mini", "--yes", "--cache-dir", str(tmp_path)])
    out = capsys.readouterr().out

    assert code == 0 and calls == ["phi-3.5-mini"]
    assert "[1/1] genai_config.json" in out
    assert "ready:" in out and "--provider local" in out

    # a second run sees the cache and does not fetch again
    assert main(["setup", "--download", "phi-3.5-mini", "--yes", "--cache-dir", str(tmp_path)]) == 0
    assert calls == ["phi-3.5-mini"]
    assert "already in" in capsys.readouterr().out


def test_setup_rejects_an_unknown_model(tmp_path: Path, capsys) -> None:
    assert main(["setup", "--download", "llama-999b", "--cache-dir", str(tmp_path)]) == 2
    assert "unknown local model" in capsys.readouterr().err
