from wyra.data_maker import FineTuningDataMaker
import json

fine_tunner = FineTuningDataMaker()

content = """
As línguas Tupi formam uma família linguística amplamente falada pelos povos indígenas da América do Sul, especialmente na região do Brasil, antes da chegada dos europeus. O nome "Tupi" refere-se tanto à família linguística quanto ao grupo de povos indígenas que falavam essas línguas.

Classificação
As línguas Tupi pertencem a uma das maiores famílias linguísticas indígenas da América do Sul, conhecida como família Tupi-Guarani, que é uma das subfamílias mais estudadas. Elas incluem diversas línguas, entre as quais:

Tupi Antigo: A língua extinta falada pelos povos Tupi no litoral brasileiro no início da colonização portuguesa.
Guarani: Uma das línguas Tupi mais conhecidas, ainda falada amplamente no Paraguai, sendo uma das línguas oficiais do país.
Nheengatu: Também conhecida como "língua geral", foi uma variação do Tupi antigo adaptada pelos colonizadores e missionários para ser uma língua franca entre as várias tribos e colonos na Amazônia e no Brasil.
História e Importância
Os povos que falavam línguas Tupi viviam ao longo da costa brasileira e em áreas do interior, espalhados por várias regiões da América do Sul. Estima-se que a chegada dos portugueses ao Brasil em 1500 encontrou cerca de 1 milhão de pessoas que falavam essas línguas, especialmente na faixa costeira.

A língua Tupi Antiga foi uma das primeiras a ser registrada pelos colonizadores portugueses. Padres jesuítas, como José de Anchieta, desempenharam um papel importante ao estudá-la e escrevê-la em textos religiosos e didáticos. O Tupi Antigo influenciou profundamente o português falado no Brasil, dando origem a muitos dos topônimos, nomes de animais, plantas e expressões que ainda estão presentes no português brasileiro.

Tupi-Guarani
A família Tupi-Guarani é composta por dezenas de línguas diferentes, muitas das quais ainda são faladas em várias partes da América do Sul, incluindo Brasil, Paraguai, Bolívia, Peru e Argentina. As línguas dessa família podem ser divididas em dois grandes ramos: as línguas Tupi (como o Tupi antigo) e as línguas Guarani (como o Guarani paraguaio).

As línguas Tupi influenciaram fortemente a cultura e a sociedade brasileira. Na fase inicial da colonização, o Tupi Antigo era falado não só pelos povos indígenas, mas também por muitos colonizadores portugueses, missionários e outros europeus que conviviam com as populações nativas.

Nheengatu (Língua Geral Amazônica)
Após a extinção do Tupi Antigo, uma versão simplificada e adaptada chamada Nheengatu continuou a ser usada em partes do Brasil, principalmente na Amazônia. O Nheengatu se tornou uma "língua geral", uma espécie de idioma comum utilizado entre diferentes povos indígenas e colonos. Hoje, ainda é falado por algumas comunidades na Amazônia.

Aspectos Linguísticos
As línguas Tupi têm uma estrutura gramatical bem diferente do português, e uma de suas características marcantes é a aglutinação, em que palavras complexas são formadas pela combinação de várias raízes e afixos. Isso faz com que palavras possam transmitir informações detalhadas sobre ações, estados e relações de maneira concisa.

Um exemplo famoso dessa característica é a palavra "Anhangabaú", de origem Tupi, que pode ser desmembrada em "anhanga" (espírito ou demônio) e "baú" (água), significando "rio do demônio".

Legado no Português Brasileiro
Muitas palavras de origem Tupi foram incorporadas ao português brasileiro, especialmente em áreas como:

Topônimos: Muitos nomes de cidades, rios, e estados brasileiros têm origem em línguas Tupi, como Pará, Paraná, Piauí, Itamaracá, Ibirapuera, entre outros.
Nomes de Animais e Plantas: Palavras como "jacaré", "tatu", "piranha", "capivara", "abacaxi", "mandioca" e "caju" são de origem Tupi.
Expressões Cotidianas: Algumas expressões comuns no português brasileiro derivam diretamente das línguas Tupi, como "curumim" (menino) e "tucupi" (um molho feito de mandioca).
"""

formatted_content = fine_tunner.format_data(content)

print(formatted_content)
with open('formatted_content.jsonl', 'w', encoding='utf-8') as file:
    file.write(formatted_content)