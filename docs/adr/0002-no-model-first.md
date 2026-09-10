# 2. Deterministic converters come first, model generation second

Date: 2026-09-09. Status: accepted.

## Context

The library's original premise was that turning text into fine-tuning data requires a
language model. That is only true for one case: free prose containing no pairs to extract.

Most sources are not that. Documentation has headings and sections. FAQs have questions and
answers. Product tables, ticket exports and database dumps have columns. Public datasets
already exist in Alpaca or ShareGPT form. In all of these the pairs are present, and a
model asked to "generate" them can only paraphrase, hallucinate or drop detail, while
costing money and producing a different result on every run.

## Decision

Deterministic converters are the primary path and the default. `generator="auto"` resolves
a converter from the file extension and never resolves one that would call a model: plain
text without an explicit generator is an error, not a silent network request.

Model-backed generation stays available for prose, with a free local option (Ollama) as
the documented default so the capability costs nothing to try.

Everything downstream of generation is code regardless of the path: schema validation,
normalization, deduplication, budgets, a seeded split and a manifest.

## Consequences

The common case is reproducible, free and offline. Two examples run without any provider
installed. The trade-off is a wider surface than a single `generate()` entry point, and a
README that has to explain when a model is warranted, which is itself the useful lesson.
