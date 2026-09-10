# wyra

Build fine-tuning datasets (JSONL) from your own data.

Most of what a dataset needs is not generation, it is **conversion and curation**, and
both are plain code. wyra treats a language model as one optional component among several:
useful when the source is free prose, unnecessary when the source already has structure.

"Wyra" means *bird* in Tupi. Like a bird, this library helps you fly through the boring
part of fine-tuning.

```bash
pip install wyra
wyra build docs/ -o dataset --valid 0.1
```

No API key, no account, no network call.

## Two ways to build a dataset

**Without a model (deterministic, free, reproducible).** Best quality, because nothing is
invented. Use it whenever the source already carries the pairs:

| Source | Generator | What becomes what |
|---|---|---|
| Markdown | `markdown` | heading becomes the question, section body becomes the answer |
| FAQ, articles, transcripts | `qa` | `Q:`/`A:` markers, or a question in prose answered by the next paragraph |
| CSV, JSON, JSONL, database dumps | `template` | one template, rendered once per row |
| Existing datasets (Alpaca, ShareGPT, chat) | `convert` | re-encoded, normalized, deduplicated |
| Any prose | `continuation`, `cloze` | self-supervised: continue the text, or fill the blank |

**With a model, when the source is free prose and no pairs exist to extract.** A local
model through [Ollama](https://ollama.com) costs nothing and needs no key; OpenAI-compatible
servers and Gemini are also supported. Every provider sits behind the same interface, so
swapping one is a constructor argument, not a rewrite.

Whichever path you take, everything after generation is deterministic code: schema
validation, normalization, deduplication, length and token budgets, a seeded split, and a
manifest recording where each example came from.

## Install

```bash
pip install wyra                 # core: no dependencies at all
pip install "wyra[tokens]"       # exact token counts with tiktoken
pip install "wyra[openai]"       # OpenAI, Azure, vLLM, LM Studio, Groq
pip install "wyra[gemini]"       # Google Gemini
pip install "wyra[all]"          # everything above
```

The Ollama adapter needs no extra: it speaks HTTP through the standard library.

## Quickstart

### Markdown into JSONL, no model

```python
from wyra import build_dataset

result = build_dataset(
    "docs/clean_code.md",
    out_dir="dataset",
    lang="pt-br",
    system_prompt="Você é um tutor de boas práticas de programação.",
    validation_fraction=0.1,
)
print(result.report.summary())   # normalize: 10/10 -> dedup: 10/10
```

### A table into thousands of examples

Write the phrasing once; every row becomes an example. This is how FLAN and T0 were built.

```python
from wyra import build_dataset
from wyra.generators import TemplateGenerator

build_dataset(
    "produtos.csv",
    out_dir="dataset",
    generator=TemplateGenerator(
        user="Qual o preço de {nome}?",
        assistant="{nome} custa R$ {preco}.",
    ),
)
```

With no templates at all, common column names are detected (`question`/`answer`,
`pergunta`/`resposta`, `instruction`/`input`/`output`).

### Free prose with a local model

```python
from wyra import build_dataset
from wyra.generators import QAPairGenerator
from wyra.prompts import PT_BR
from wyra.providers import create

generator = QAPairGenerator(create("ollama", model="llama3.2:3b"), prompts=PT_BR, per_chunk=3)
build_dataset("notas/*.txt", out_dir="dataset", generator=generator, validation_fraction=0.1)
```

The same call with Gemini, choosing the provider from the environment:

```bash
export WYRA_PROVIDER=gemini GEMINI_API_KEY=...
```

```python
build_dataset("notas/*.txt", out_dir="dataset", generator="llm-qa")
```

### Validate and convert what you already have

```python
from wyra import convert_jsonl, validate_jsonl

report = validate_jsonl("data/train.jsonl")
print(report.summary())
assert report.ok

convert_jsonl("data/sharegpt.jsonl", "data/train.jsonl", output_format="openai-chat")
```

## Command line

```bash
wyra build docs/*.md -o dataset --lang pt-br --valid 0.1 --system "Seja um tutor."
wyra build faq.txt   -o dataset --generator qa
wyra build rows.csv  -o dataset --generator template --user "{pergunta}" --assistant "{resposta}"
wyra build notas.txt -o dataset --generator llm-qa --provider ollama --model llama3.2:3b
wyra validate dataset/train.jsonl
wyra convert dataset/train.jsonl -o alpaca.jsonl --to alpaca
```

`wyra validate` exits non-zero when any record is invalid, so it drops straight into CI.

## Output formats

`openai-chat` (the default, also used by Azure OpenAI, TRL, Axolotl and Unsloth), `alpaca`
and `sharegpt`. Pick one with `--format` or `output_format=`.

Validation follows the OpenAI cookbook rules exactly, and reports every problem instead of
stopping at the first: unknown message keys, unrecognized roles, empty content, a missing
assistant message, a conversation that does not end with the assistant, and invalid JSON,
each with the line number.

## The manifest

Every build writes `manifest.json` next to the data. It is the dataset's lineage, produced
as a by-product rather than as paperwork:

```json
{
  "wyra_version": "0.1.0",
  "generator": {"name": "llm-qa", "params": {"model": "llama3.2:3b", "prompt_sha256": "d36f…"}},
  "sources": [{"source": "docs/clean_code.md", "sha256": "9c1a…", "chars": 11866}],
  "curation": [{"step": "normalize", "in": 17, "out": 17}, {"step": "dedup", "in": 17, "out": 15}],
  "counts": {"generated": 17, "kept": 15, "train": 14, "validation": 1, "errors": 0},
  "split": {"validation_fraction": 0.1, "seed": 42},
  "stats": {"token_counter": "approx", "tokens": {"min": 42, "max": 116, "mean": 68.5}}
}
```

Same sources plus same seed give the same split, in any process.

## Configuration

Configuration comes from the environment. Nothing is read from a file and no credential is
ever stored in code.

| Variable | Meaning |
|---|---|
| `WYRA_PROVIDER` | `ollama` (default), `openai` or `gemini` |
| `WYRA_MODEL` | model name, overriding the provider's default |
| `OLLAMA_HOST` | defaults to `http://localhost:11434` |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL` | the base URL also reaches Azure, vLLM, LM Studio and Groq |
| `GEMINI_API_KEY` | `GOOGLE_API_KEY` is accepted too |

A missing key fails immediately, with the exact variable to export, before any request.

## Extending it

Every seam is a `typing.Protocol` you can implement. Nothing needs a subclass except the
LLM generator, which shares real implementation.

```python
from wyra import Document, Example, build_dataset

class HeadlineGenerator:
    name = "headlines"

    def describe(self):          # goes into the manifest
        return {"style": "news"}

    def generate(self, doc: Document):
        for line in doc.text.splitlines():
            if line.strip():
                yield Example.qa("Write a headline for this text.", line, source=doc.ref)

build_dataset("news.txt", out_dir="dataset", generator=HeadlineGenerator())
```

Ports: `ExampleGenerator`, `CompletionProvider`, `Chunker`, `TokenCounter`, `CurationStep`,
`DatasetWriter`. Register a generator or a provider by name with
`wyra.generators.register` or `wyra.providers.register`.

Testing your own generator needs no network and no key:

```python
from wyra.providers import FakeCompletionProvider

provider = FakeCompletionProvider(responses=['{"pairs": [{"question": "q?", "answer": "a"}]}'])
```

## What this library will not do for you

- **Fine-tuning teaches form, not facts.** Structured output, domain vocabulary, tone and
  refusal behaviour respond well to it. If the problem is "the model does not know our
  documents", the answer is retrieval, not weights.
- **A schema check is not a quality check.** Validation guarantees the shape of every
  record and never its meaning. Read a sample before you train, especially when a small
  local model wrote it.
- **Small models write weak pairs.** The grounding filter and deduplication remove the
  worst, not the mediocre.

## Migration from 0.0.x

`FineTuningDataMaker` and `CryptoHandler` are gone in 0.1.0. The old
`format_data(text)` call becomes:

```python
from wyra import build_dataset
from wyra.generators import QAPairGenerator
from wyra.providers import create

build_dataset(source, out_dir="dataset", generator=QAPairGenerator(create("gemini")))
```

The API key now comes from `GEMINI_API_KEY`. **Versions up to 0.0.6 contained a hardcoded
API key in the published package. If you ever installed one of them, that key is
compromised and should be treated as such by whoever owns it.**

## License

MIT. See [LICENSE](LICENSE).
