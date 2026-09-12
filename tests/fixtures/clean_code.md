# Código limpo

Resumo curto do livro Clean Code, de Robert C. Martin, com as ideias que mais mudam o dia a dia.

## Nomes são muito importantes

Nomear variáveis, funções e classes de acordo com o que fazem é essencial para o entendimento do código. Métodos devem ter nomes de verbos e classes devem usar substantivos. Não tenha medo de nomes grandes quando eles descrevem melhor a intenção.

## Funções pequenas

A primeira regra das funções é que elas precisam ser pequenas. A segunda regra é que elas têm de ser ainda menores. Uma função faz uma coisa só e a faz bem.

```python
# um "#" dentro de uma cerca de código não é cabeçalho
def soma(a, b):
    return a + b
```

### Por que isso acontece?

A gente sempre tem alguma desculpa: o prazo era curto, os requisitos mudaram. Mas a culpa é nossa, e cabe ao desenvolvedor defender a importância de escrever código limpo.

## Curta

Seção curta demais.

## Regra de Escoteiro

Deixe o código mais limpo do que estava antes de você mexer nele. Refatorar deve ser um hábito, feito enquanto a lógica ainda está fresca na cabeça. Pequenas ações, como renomear um método confuso, já contam.
