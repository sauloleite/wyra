"""A product table into thousands of examples: write the phrasing once, render every row.

This is the most productive model-free path. Each template produces one example per row,
so four products and three templates already give twelve examples.

    python examples/build_template.py
"""

from __future__ import annotations

from pathlib import Path

from wyra import build_examples, read_documents
from wyra.generators import TemplateGenerator

HERE = Path(__file__).parent
SYSTEM = "Você é o atendimento de uma loja. Responda em português, de forma curta."

TEMPLATES = [
    ("Qual o preço de {nome}?", "{nome} custa R$ {preco}."),
    ("Em quanto tempo o {nome} chega?", "O prazo de entrega do {nome} é de {prazo}."),
    (
        "Em que categoria está {nome}?",
        "{nome} está na categoria {categoria} e custa R$ {preco}.",
    ),
]

documents = read_documents(HERE / "data" / "produtos.csv")
examples = [
    example
    for user, assistant in TEMPLATES
    for document in documents
    for example in TemplateGenerator(user, assistant, system=SYSTEM).generate(document)
]

result = build_examples(
    examples, HERE / "out" / "template" / "train.jsonl", validation_fraction=0.2
)
print(result.report.summary())
print(
    f"{result.counts['kept']} examples from {len(documents)} file(s) and {len(TEMPLATES)} templates"
)
print(f"train: {result.train_path}")
