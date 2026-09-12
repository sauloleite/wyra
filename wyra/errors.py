"""Exception hierarchy. Leaf module: it imports nothing from the package."""

from __future__ import annotations


class WyraError(Exception):
    """Base class for every error raised by wyra."""


class ConfigError(WyraError):
    """Missing or invalid configuration: env var, provider name, optional extra."""


class ProviderError(WyraError):
    """A completion provider failed at the transport or SDK level."""


class GenerationError(WyraError):
    """Model output could not be turned into examples (not JSON, wrong shape, truncated)."""


class TruncatedOutputError(GenerationError):
    """The model hit its output limit before finishing.

    Distinct from other generation failures because the remedy is different: ask for fewer
    items, or give the model more room, rather than retrying the same request.
    """


class ValidationError(WyraError):
    """A record or message violates the dataset schema.

    ``issues`` carries machine-readable tags (the OpenAI cookbook categories plus a few of
    our own) and ``index`` the 1-based line number when the record came from a file.
    """

    def __init__(
        self, message: str, *, issues: tuple[str, ...] = (), index: int | None = None
    ) -> None:
        super().__init__(message)
        self.issues = issues
        self.index = index


class FormatError(WyraError):
    """Unknown or undetectable dataset format, or a record a format cannot represent."""
