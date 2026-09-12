"""Instruction phrasing used by the generators, with presets per language.

Prompts are data, not code: a preset is a frozen record and the generators only fill the
placeholders. ``by_lang`` is the single lookup point.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import ConfigError


@dataclass(frozen=True, slots=True)
class Prompts:
    lang: str
    explain_section: str
    explain_section_in: str
    continue_text: str
    fill_blank: str
    qa_system: str
    qa_task: str
    instruction_task: str
    json_only: str


EN = Prompts(
    lang="en",
    explain_section="Explain: {title}.",
    explain_section_in="In the context of {parent}, explain: {title}.",
    continue_text="Continue the following text.\n\n{context}",
    fill_blank="Fill in the blank marked ____ in the following text.\n\n{text}",
    qa_system=(
        "You write training data. Using only the text provided, produce faithful "
        "question/answer pairs. Never invent facts that are not in the text."
    ),
    qa_task=(
        "Based only on the text below, write {n} self-contained question/answer pairs in the "
        "same language as the text. Questions must be answerable from the text; answers must "
        "be complete sentences.\n\nText:\n{text}"
    ),
    instruction_task=(
        "Based only on the text below, write {n} instruction-tuning items in the same "
        "language as the text. Each item has an instruction, an optional input, and the "
        "output a helpful assistant would give.\n\nText:\n{text}"
    ),
    json_only="Your previous reply was not valid JSON. Reply with the JSON object only.",
)

PT_BR = Prompts(
    lang="pt-br",
    explain_section="Explique: {title}.",
    explain_section_in="No contexto de {parent}, explique: {title}.",
    continue_text="Continue o texto a seguir.\n\n{context}",
    fill_blank="Preencha a lacuna marcada com ____ no texto a seguir.\n\n{text}",
    qa_system=(
        "Você escreve dados de treinamento. Usando apenas o texto fornecido, produza pares de "
        "pergunta e resposta fiéis ao texto. Nunca invente fatos que não estejam no texto."
    ),
    qa_task=(
        "Com base apenas no texto abaixo, escreva {n} pares de pergunta e resposta "
        "autocontidos, no mesmo idioma do texto. As perguntas devem ser respondíveis pelo "
        "texto; as respostas devem ser frases completas.\n\nTexto:\n{text}"
    ),
    instruction_task=(
        "Com base apenas no texto abaixo, escreva {n} itens de instruction tuning no mesmo "
        "idioma do texto. Cada item tem uma instrução, uma entrada opcional e a saída que um "
        "assistente prestativo daria.\n\nTexto:\n{text}"
    ),
    json_only="Sua resposta anterior não era JSON válido. Responda apenas com o objeto JSON.",
)

PRESETS: dict[str, Prompts] = {"en": EN, "pt-br": PT_BR, "pt": PT_BR, "pt_br": PT_BR}


def by_lang(lang: str) -> Prompts:
    try:
        return PRESETS[lang.lower()]
    except KeyError:
        known = ", ".join(sorted(PRESETS))
        raise ConfigError(f"unknown language {lang!r}; known: {known}") from None
