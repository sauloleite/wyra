from wyra.data_maker import FineTuningDataMaker
import json

fine_tunner = FineTuningDataMaker()

content = """
Como programadores, a nossa primeira prioridade é criar código que funciona. Infelizmente, código que simplesmente “funciona” não é suficiente. Código que tem valor real e é duradouro, tem que ser “limpo”. Em Clean Code: A Handbook of Agile Software Craftsmanship traduzido para o português como Código Limpo: Habilidades Práticas do Agile Software, de Robert C. Martin ou “Uncle Bob” como ele se auto intitula, usa vários exemplos e casos de estudo para nos ajudar a identificar código que pode ser melhorado, e nos dá uma variedades de técnicas para fazermos essa limpeza no código.



O livro começa com uma citação do escritor, sendo muito direta quanto a razão ao qual os leitores estão buscando ao ler seu livro.

“Você está lendo esse livro por duas razões. Primeiro, você é um programador. Segundo, você quer ser um programador melhor. Ótimo. Precisamos de programadores melhores”.

A principal proposta do Clean Code é ajudar o leitor a escrever um bom código. Perceber as diferenças entre um código ruim e um código bom, bem como transformar um código ruim em um código bom.

Muitos desenvolvedores devem se fazer as perguntas abaixo:
Mas se o meu código “ruim” está funcionando, por que eu vou querer me dar ao trabalho de refatorar tudo pra deixar ele “bom”? Será que vale o trabalho?

A resposta é simples: Porque o custo total para manter um código ruim é enorme.

Quem trabalha ou trabalhou como programador por algum tempo, já deve ter passado pela experiência de ter uma produtividade ruim por causa de código ruim. Algo que deveria ter levado horas para ser feito, acabou levando semanas. Uma mudança que bastava ser em poucas linhas de código, acabou precisando ser feita também em muitos outros lugares do sistema e estes precisaram ser bem entendidos para alterá-los.

No início de um projeto, sempre se começa com uma alta produtividade, porém com o tempo, muitos projetos acabam reduzindo drasticamente a produtividade.

Por que isso acontece?
A gente sempre tem alguma desculpa pra dar: o prazo de entrega era muito curto; os requisitos mudaram no meio do projeto; o gerente de produtos pediu coisas absurdas, entre outras. Mas a culpa na verdade é nossa. Nós como desenvolvedores devemos ser profissionais a ponto de defender a importância de escrever um código limpo, mesmo que demore mais pra ser feito.

Tem um exemplo muito bom no livro que diz o seguinte:

“E se você fosse um médico e tivesse um paciente que pediu que você parasse com toda essa besteira de lavar as mãos antes de uma cirurgia pois estava demorando demais? Claramente o paciente é o chefe, mas o médico claramente deve negar o pedido do paciente. Por quê? Porque o médico sabe mais do que o paciente sobre os riscos de doenças e infecções. Seria uma atitude não profissional (sem falar criminal) se o médico aceitasse o pedido.”

Seguindo a analogia, também não é profissional que um desenvolvedor se curve a todos os desejos de um gerente, que não entende os riscos de um código ruim. Os gerentes vão defender o prazo de entrega com unhas e dentes, mas isso faz parte do trabalho deles! Nosso trabalho é defender um código limpo com a mesma intensidade.

O livro também mostra como programadores experientes interpretam o que seria um código limpo. Entre eles, o autor do livro (Uncle Bob), descreve que para ele um código limpo é um código que não precisa ser explicado. Um código que qualquer programador possa bater o olho e saber exatamente o que ele faz.

“Qualquer tolo pode escrever um código que um computador consiga entender. Bons programadores escrevem código que humanos consigam entender.”

- Martin Fowler

Mas também não basta saber escrever um código limpo. É preciso manter o código limpo com o passar do tempo também, porque se tem uma coisa que é bastante comum de acontecer é começar um projeto bonitinho do zero, mas com o passar do tempo ele virar uma completa bagunça.

O livro está divido em três partes. Na primeira há diversos capítulos que descrevem os princípios, padrões e práticas para criar um código limpo.

A segunda parte consiste em diversos casos de estudo de complexidade cada vez maior. Cada um é um exercício para limpar um código — transformar o código base que possui alguns problemas em um melhor e eficiente. A terceira parte é a compensação: um único capítulo com uma lista de heurísticas e “odores” reunidos durante a criação dos estudos de caso. O resultado será um conhecimento base que descreve a forma como pensamos quando criamos, lemos e limpamos um código.

Nomes são muito importantes
É comum as pessoas serem apelidadas pelas suas características mais comuns. Para termos um código limpo, devemos fazer isso com ele. Nomear variáveis, funções, parâmetros, classes ou métodos de acordo com suas funcionalidades. Isso é essencial para um bom entendimento do código.

Ao definir um nome, precisamos ter em mente dois pontos principais:

Ser preciso: precisamos passar a ideia central da nossa variável ou método, sem dar voltas, sendo conciso e direto.
Não ter medo de nomes grandes: um nome bem descritivo, mesmo que seja grande, irá possibilitar uma melhor compreensão e posterior manutenção do código.
Para finalizar, é recomendável que:

Métodos ou Funções: devem ter nome de verbos, para assim, expressar quais são suas finalidades;
Classes e Objetos: deve ser utilizado substantivos.
Seja um verdadeiro autor do seu código

O código é uma história. Então, como um bom autor, devemos nos preocupar com a maneira de como contar essa história. A ideia desse tópico é simples! Para estruturar um código limpo, é necessário criar funções simples, pequenas e claras. Segundo Robert, a primeira regra das funções é a seguinte:

“Elas precisam ser pequenas.”

Já a segunda regra das funções diz o seguinte:

“Elas têm de ser ainda menores.”

As funções devem ter apenas uma tarefa e, assim, saber cumpri-la da maneira mais simples possível. Isso possibilita que um método seja reutilizado diversas vezes em seu código, facilitando sua manutenção à longo prazo.

Para ajudar a vida dos programadores, o paradigma de programação orientado a objetos auxilia na criação de métodos que podem ser utilizados em todas as funcionalidades do software.

Comente. Mas só o necessário!
Comente o necessário e somente o necessário. Códigos são constantemente modificados, enquanto comentários, raramente. Assim, é comum um comentário deixar de ter significado, ou pior ainda, passar um significado falso depois de algum tempo.

Além disso, códigos com muitos comentários são tão ruins que, com o tempo, nossos olhos acabam ignorando todos. Então, o melhor não é comentar os códigos ruins e sim reescrevê-los.

Utilize DRY
DRY é o anacrônico para Don’t repeat yourself (Não repita a si mesmo). É o conceito que diz que cada parte de conhecimento do sistema deve possuir apenas uma representação. Desta forma, evitando a ambiguidade do código. Em outras palavras, não deve existir duas partes do programa que desempenham a mesma função, ou seja, o famoso copiar e colar no código.

Mas porque evitar repetição? Simples!

Quem tem uma segunda casa na praia, ou no campo, sabe o quão complicado é garantir a manutenção das duas. Mesmo que a repetição possa parecer inofensiva em programas mais simples, ela pode vir a ser um problema à medida que o software vai crescendo e as manutenções e desenvolvimentos se tornam cada vez mais complexos.

Uma boa maneira de evitar a duplicidade do código é aplicar corretamente a técnica de responsabilidade única. Para cada função ou método, utilizar apenas uma parte do método (ou função). O correto é abstrair apenas essa parte e criar um novo!

Melhor prevenir do que remediar…

Esse famoso ditado se aplica ao desenvolvimento de software também. Bons desenvolvedores pensam que as coisas podem dar errado, pois isso eventualmente irá acontecer. Desta forma, o código deve estar preparado para lidar com esses problemas que surgirão.

Hoje a maioria das linguagens possuem recursos para tratar erros nos códigos através de Exceptions e blocos try-catch.

Exceptions: mecanismo que sinaliza eventos excepcionais. Por exemplo, tentar inserir o caractere “a” em uma variável do tipo inteiro;
Blocos try-catch: capturam as exceções citadas. Portanto, devem ser utilizados de maneira global. Afinal de contas, os métodos já possuem suas funções (que não é tratar erros).
Para finalizarmos esse tópico, uma dica excelente para não gerar erros em seu código é simplesmente não utilizar “null”, tanto como parâmetro, quanto para retorno em funções. Muitas vezes, esses retornos exigem verificações desnecessárias que, caso não sejam feitas, podem gerar erros.

Regra de Escoteiro

Os escoteiros possuem uma regra: deixe o acampamento mais limpo do que você o encontrou. Para desenvolvedores, podemos adaptar para:

“Deixe o código mais limpo do que estava antes de você mexer nele.”

É uma regra simples, mas bastante efetiva. Quando você estiver trabalhando em algum código, se encontrou algo que pode ser melhorado, melhore! Não deixe pra depois, porque o depois na verdade quer dizer nunca.

Você não precisa refatorar a classe inteira, são ações pequenas como mudar o nome de um método que não estava muito claro ou quebrar um método que estava muito grande em outros menores.

Refatorar o código deve ser um hábito. É uma atividade essencial para a modernização do software e deve ser feita com muito cuidado para não impactar as funcionalidades existentes. Pensamentos como “isso é perda de tempo” ou “se tá funcionando melhor não mexer” são prejudiciais a longo prazo. O melhor é refatorar o mais rápido possível enquanto a lógica ainda está fresca na cabeça.

Testes limpos
Um código só está realmente limpo se ele for validado. Mas a questão é, como é possível manter um teste limpo? A resposta é simples, da mesma maneira que mantemos o nosso código limpo, com clareza, simplicidade e consistência de expressão.

Testes limpos seguem as regras do anacrônico FIRST (Fast, Indepedent, Repeatable, Self-validation, Timely).

Rapidez: os testes devem ser rápidos para que possam ser executados diversas vezes;
Independência: quando testes são dependentes, uma falha pode causar um efeito dominó dificultando a análise individual;
Repetitividade: deve ser possível repetir o teste em qualquer ambiente;
Auto validação: bons testes possuem como resultado respostas do tipo “verdadeiro” ou “falso”. Caso contrário, a falha pode se tornar subjetiva;
Pontualidade: os testes precisam ser escritos antes do código de produção, onde os testes serão aplicados. Caso contrário, o código pode ficar complexo demais para ser testado ou até pode ser que o código não possa ser testado.
Tenha orgulho do seu código!
Acima falamos sobre vários conceitos de Clean Code, mas Michael Feathers (autor do livro Legacy Code) conseguiu resumir Código Limpo em uma frase que resume tudo!

“Um código limpo sempre parece que foi escrito por alguém que se importava”

Conclusão:
Se importar com um código é tomar cuidado na hora de:

Dar nomes;
Criar funções pequenas e especificas;
Não ser redundante;
Estar sempre atento a erros;
Não deixar de refatorar;
Utilizar testes limpos para validações.
No fim, a sensação de orgulho é recompensadora! O que compensa todo o trabalho duro durante o desenvolvimento!

Na minha opinião o livro é de leitura fundamental para todos os desenvolvedores, uma vez que ele nos faz refletir sobre nossas decisões na hora de programar e os aprendizados adquiridos nele são levados adiante, como se fossemos evangelizados a seguir boas práticas de desenvolvimento."""

formatted_content = fine_tunner.format_data(content)

print(formatted_content)
with open('formatted_content.jsonl', 'w', encoding='utf-8') as file:
    file.write(formatted_content)