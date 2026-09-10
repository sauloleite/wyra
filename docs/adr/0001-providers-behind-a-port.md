# 1. Model providers sit behind a port we define

Date: 2026-09-09. Status: accepted.

## Context

Version 0.0.6 called the Google Gemini SDK directly from `FineTuningDataMaker`. Three
consequences followed. The SDK's request and response types were part of our public
behaviour, so a provider change was a rewrite. Tests had to mock SDK internals, so they
broke whenever the SDK moved and proved nothing about our code. And the API key was read
at construction time from a literal in the source, which is how it ended up published on
PyPI.

The model is the most expensive and least stable component of this system. It is exactly
the piece that must not be wired directly into the core.

## Decision

The core defines `CompletionProvider`: a `Protocol` taking our `Message` values and
returning our `Completion` value. Adapters in `wyra/providers/` translate to and from each
SDK, import that SDK lazily inside their own factory, and convert every SDK exception into
`ProviderError`. Credentials are read from the environment by `Settings.from_env`, and a
missing one raises `ConfigError` before any request is made.

`FakeCompletionProvider` ships with the library, not with the tests, because anyone
building on wyra needs the same test double we do.

## Consequences

The core declares no dependencies. Adding a provider is a new module plus one registry
entry, with no change to the generators or the builder. Every test runs offline. The cost
is one translation layer per provider and a `Completion` that exposes less than each SDK's
native response; both are worth it.
