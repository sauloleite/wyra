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
