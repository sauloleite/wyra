"""wyra: build fine-tuning datasets (JSONL) from your own data.

Deterministic converters first (Markdown sections, FAQ pairs, tables with templates,
existing datasets, self-supervised continuation and cloze); local or cloud LLMs only
when the source is free prose, and always behind a port you can fake in tests.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .builder import BuildResult, DatasetBuilder, split_examples
from .domain import Completion, Document, Example, Message, Role
from .errors import (
    ConfigError,
    FormatError,
    GenerationError,
    ProviderError,
    TruncatedOutputError,
    ValidationError,
    WyraError,
)
from .pipeline import build_dataset, build_examples, convert_jsonl
from .readers import read_documents, read_examples
from .validation import ValidationReport, validate_jsonl

__all__ = [
    "BuildResult",
    "Completion",
    "ConfigError",
    "DatasetBuilder",
    "Document",
    "Example",
    "FormatError",
    "GenerationError",
    "Message",
    "ProviderError",
    "Role",
    "TruncatedOutputError",
    "ValidationError",
    "ValidationReport",
    "WyraError",
    "__version__",
    "build_dataset",
    "build_examples",
    "convert_jsonl",
    "read_documents",
    "read_examples",
    "split_examples",
    "validate_jsonl",
]
