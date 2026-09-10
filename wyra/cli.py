"""Command line: ``wyra build``, ``wyra validate``, ``wyra convert``.

Thin argument parsing over the facade. Every default is explicit so the manifest can
record it, and nothing here reaches a network unless the chosen generator asks for it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from typing import Any

from . import __version__
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
    build.add_argument("--no-dedup", dest="dedupe", action="store_false")
    build.add_argument("--on-error", choices=("skip", "raise"), default="skip")
    build.add_argument("--lineage", dest="write_lineage", action="store_true")

    validate = subparsers.add_parser("validate", help="check an existing JSONL dataset")
    validate.add_argument("path")
    validate.add_argument(
        "-f", "--format", dest="input_format", default="auto", choices=["auto", *sorted(FORMATS)]
    )
    validate.add_argument("--budget", type=int, default=16385, help="per-example token budget")

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
    report = validate_jsonl(args.path, input_format=args.input_format, budget=args.budget)
    print(report.summary())
    return EXIT_OK if report.ok else EXIT_INVALID


def _convert(args: argparse.Namespace) -> int:
    result = convert_jsonl(
        args.path,
        args.out,
        input_format=args.input_format,
        output_format=args.output_format,
        dedupe=args.dedupe,
    )
    print(f"wrote {result.counts['kept']} record(s) to {args.out}")
    return EXIT_OK if result.counts["kept"] else EXIT_INVALID


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
