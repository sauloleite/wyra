"""wyra: build fine-tuning datasets (JSONL) from your own data.

Deterministic converters first (Markdown sections, FAQ pairs, tables with templates,
existing datasets, self-supervised continuation and cloze); local or cloud LLMs only
when the source is free prose, and always behind a port you can fake in tests.
"""

from __future__ import annotations

from .domain import Completion, Document, Example, Message, Role
from .errors import (
    ConfigError,
    FormatError,
    GenerationError,
    ProviderError,
    ValidationError,
    WyraError,
)

__version__ = "0.1.0"

__all__ = [
    "Completion",
    "ConfigError",
    "Document",
    "Example",
    "FormatError",
    "GenerationError",
    "Message",
    "ProviderError",
    "Role",
    "ValidationError",
    "WyraError",
    "__version__",
]
