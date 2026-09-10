from __future__ import annotations

from pathlib import Path

import pytest

from wyra.domain import Example

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES


@pytest.fixture
def sample_examples() -> list[Example]:
    return [
        Example.qa("O que é DRY?", "Não repita a si mesmo.", system="Você é um tutor."),
        Example.qa("What is SOLID?", "Five design principles."),
        Example.qa("Qual a regra do escoteiro?", "Deixe o código mais limpo do que encontrou."),
    ]
