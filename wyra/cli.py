"""Command line: ``wyra build``, ``wyra validate``, ``wyra convert``.

Thin argument parsing over the facade. Every default is explicit so the manifest can
record it, and nothing here reaches a network unless the chosen generator asks for it.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from collections.abc import Sequence
from typing import Any

from . import __version__
from .config import Settings
from .errors import WyraError
from .formats import FORMATS
from .generators import available as available_generators
from .pipeline import build_dataset, convert_jsonl
from .providers import available as available_providers
from .validation import validate_jsonl

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wyra",
        description="Build fine-tuning datasets (JSONL) from your own data.",
    )
    parser.add_argument("--version", action="version", version=f"wyra {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="log what is happening")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="generate a dataset from sources")
    build.add_argument("sources", nargs="+", help="files, folders or globs")
    build.add_argument("-o", "--out", required=True, help="output directory")
    build.add_argument(
        "-g",
        "--generator",
        default="auto",
        help=f"auto or one of: {', '.join(available_generators())}",
    )
    build.add_argument(
        "-p", "--provider", default=None, help=f"one of: {', '.join(available_providers())}"
    )
    build.add_argument("-m", "--model", default=None, help="model name for the provider")
    build.add_argument("--lang", default="en", help="prompt language: en or pt-br")
    build.add_argument(
        "-f", "--format", dest="output_format", default="openai-chat", choices=sorted(FORMATS)
    )
    build.add_argument("--system", default=None, help="system prompt added to every example")
    build.add_argument("--user", default=None, help="user template for the 'template' generator")
    build.add_argument("--assistant", default=None, help="assistant template for 'template'")
    build.add_argument(
        "--valid", dest="validation_fraction", type=float, default=0.0, metavar="FRACTION"
    )
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--per-chunk", type=int, default=None, help="examples per chunk (LLM)")
    build.add_argument("--max-tokens", type=int, default=None, help="drop examples above this")
    build.add_argument(
        "--tokens",
        dest="token_counter",
        default="approx",
        help="how to count tokens: approx (default) or tiktoken, which needs wyra[tokens]",
    )
    build.add_argument("--no-dedup", dest="dedupe", action="store_false")
    build.add_argument("--on-error", choices=("skip", "raise"), default="skip")
    build.add_argument("--lineage", dest="write_lineage", action="store_true")

    validate = subparsers.add_parser("validate", help="check an existing JSONL dataset")
    validate.add_argument("path")
    validate.add_argument(
        "-f", "--format", dest="input_format", default="auto", choices=["auto", *sorted(FORMATS)]
    )
    validate.add_argument("--budget", type=int, default=16385, help="per-example token budget")
    validate.add_argument(
        "--tokens",
        dest="token_counter",
        default="approx",
        help="how to count tokens: approx (default) or tiktoken, which needs wyra[tokens]",
    )

    convert = subparsers.add_parser("convert", help="re-encode a dataset into another format")
    convert.add_argument("path")
    convert.add_argument("-o", "--out", required=True, help="output file")
    convert.add_argument(
        "--from", dest="input_format", default="auto", choices=["auto", *sorted(FORMATS)]
    )
    convert.add_argument(
        "--to", dest="output_format", default="openai-chat", choices=sorted(FORMATS)
    )
    convert.add_argument("--no-dedup", dest="dedupe", action="store_false")
    convert.add_argument(
        "--skip-invalid",
        dest="on_invalid",
        action="store_const",
        const="skip",
        default="raise",
        help="drop records that do not parse instead of stopping",
    )

    setup = subparsers.add_parser(
        "setup", help="report what is installed and available, and fetch a local model"
    )
    setup.add_argument(
        "--download", metavar="MODEL", default=None, help="fetch one catalogue model"
    )
    setup.add_argument("--yes", action="store_true", help="do not ask before downloading")
    setup.add_argument("--cache-dir", default=None, help="where to keep the weights")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )
    try:
        if args.command == "build":
            return _build(args)
        if args.command == "validate":
            return _validate(args)
        if args.command == "setup":
            return _setup(args)
        return _convert(args)
    except WyraError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE


def _build(args: argparse.Namespace) -> int:
    extra: dict[str, Any] = {}
    if args.user or args.assistant:
        extra["user"] = args.user
        extra["assistant"] = args.assistant
    if args.per_chunk is not None:
        extra["per_chunk"] = args.per_chunk

    result = build_dataset(
        args.sources,
        args.out,
        generator=args.generator,
        provider=args.provider,
        model=args.model,
        lang=args.lang,
        output_format=args.output_format,
        system_prompt=args.system,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        max_tokens=args.max_tokens,
        dedupe=args.dedupe,
        token_counter=args.token_counter,
        on_error=args.on_error,
        write_lineage=args.write_lineage,
        **extra,
    )
    counts = result.counts
    print(
        f"generated {counts['generated']}, kept {counts['kept']}, "
        f"train {counts['train']}, validation {counts['validation']}"
    )
    if counts["errors"]:
        print(f"skipped {counts['errors']} chunk(s); see manifest.json", file=sys.stderr)
    print(result.report.summary())
    for path in (result.train_path, result.validation_path, result.manifest_path):
        if path is not None:
            print(f"wrote {path}")
    return EXIT_OK if counts["kept"] else EXIT_INVALID


def _validate(args: argparse.Namespace) -> int:
    report = validate_jsonl(
        args.path,
        input_format=args.input_format,
        budget=args.budget,
        counter=args.token_counter,
    )
    print(report.summary())
    return EXIT_OK if report.ok else EXIT_INVALID


def _setup(args: argparse.Namespace) -> int:
    from .providers import modelstore
    from .providers.ollama import MIN_SCHEMA_VERSION, probe, supports_schema

    if args.download:
        return _download_model(args, modelstore)

    print(f"wyra {__version__} on Python {sys.version.split()[0]}")
    print()
    print("core: installed, no dependencies")
    print(f"  generators that need no model: {', '.join(_model_free_generators())}")
    print()

    print("optional extras")
    for extra, module, purpose in (
        ("tokens", "tiktoken", "exact token counts, via --tokens tiktoken"),
        ("openai", "openai", "OpenAI, Azure, vLLM, LM Studio, Groq"),
        ("gemini", "google.genai", "Google Gemini"),
        ("local", "onnxruntime_genai", "embedded model, no daemon and no key"),
    ):
        state = "installed" if _installed(module) else f"missing: pip install 'wyra[{extra}]'"
        print(f"  {extra:7} {purpose:45} {state}")
    print()

    print("providers")
    info = probe()
    if info is None:
        print("  ollama   not reachable at the configured host (install it, or 'ollama serve')")
    else:
        models = ", ".join(info["models"]) or "no models pulled yet: 'ollama pull llama3.2:3b'"
        print(f"  ollama   version {info['version']} at {info['host']}")
        if not supports_schema(info["version"]):
            print(
                f"           too old for JSON schemas: upgrade to {MIN_SCHEMA_VERSION} "
                "or newer, or generation will fail"
            )
        print(f"           {models}")
    settings = Settings.from_env()
    for name, present in (
        ("openai", bool(settings.openai_api_key or settings.openai_base_url)),
        ("gemini", bool(settings.gemini_api_key)),
    ):
        print(f"  {name:8} credentials {'set' if present else 'not set'} in the environment")
    print()

    print(f"embedded models (cache: {modelstore.cache_root(args.cache_dir)})")
    for row in modelstore.catalog_rows(args.cache_dir):
        flags = ["default"] if row["default"] else []
        flags.append("cached" if row["cached"] else "not downloaded")
        licence = row["license"] if row["license"] != modelstore.UNDECLARED else "NO LICENCE"
        print(
            f"  {row['name']:14} {row['parameters']:>5}  {row['size_mb']:>5} MB  "
            f"{licence:11} {'  '.join(flags)}"
        )
        print(f"                 {row['note']}")
    print()
    print("to fetch one: wyra setup --download <model>")
    return EXIT_OK


def _download_model(args: argparse.Namespace, modelstore: Any) -> int:
    entry = modelstore.get_entry(args.download)
    target = modelstore.model_dir(entry.name, args.cache_dir)
    if modelstore.is_cached(entry.name, args.cache_dir):
        print(f"{entry.name} is already in {target}")
        return EXIT_OK

    print(f"model:       {entry.name} ({entry.parameters})")
    print(f"source:      {entry.url}")
    print(f"download:    {entry.size_mb} MB")
    print(f"licence:     {entry.license}")
    print(f"destination: {target}")
    if not entry.license_declared:
        print("warning: this repository declares no licence; check it before using the output.")

    if not args.yes:
        if not sys.stdin.isatty():
            print(
                "refusing to download without confirmation; pass --yes to accept",
                file=sys.stderr,
            )
            return EXIT_INVALID
        if input("download now? [y/N] ").strip().lower() not in ("y", "yes"):
            print("cancelled")
            return EXIT_OK

    def progress(name: str, index: int, total: int) -> None:
        print(f"  [{index}/{total}] {name}", flush=True)

    path = modelstore.fetch(entry, cache_dir=args.cache_dir, progress=progress)
    print(f"ready: {path}")
    print("use it with: wyra build FILE -o OUT --generator llm-qa --provider local")
    return EXIT_OK


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _model_free_generators() -> list[str]:
    from .generators import NEEDS_PROVIDER
    from .generators import available as available_generators

    return [name for name in available_generators() if name not in NEEDS_PROVIDER]


def _convert(args: argparse.Namespace) -> int:
    result = convert_jsonl(
        args.path,
        args.out,
        input_format=args.input_format,
        output_format=args.output_format,
        dedupe=args.dedupe,
        on_invalid=args.on_invalid,
    )
    print(f"wrote {result.counts['kept']} record(s) to {args.out}")
    return EXIT_OK if result.counts["kept"] else EXIT_INVALID


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
