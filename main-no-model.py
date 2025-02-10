from transformers import pipeline
import requests
from bs4 import BeautifulSoup

def extraindo_text_da_web(url):
    respose = requests.get(url)
    respose.raise_for_status()
    soup = BeautifulSoup(respose.text, "html.parser")

    paragraphs = soup.find_all("p")
    text = " ".join([para.get_text() for para in paragraphs])
    return text

url = "https://pt.wikipedia.org/wiki/Microsoft_Office"

documento = extraindo_text_da_web(url)

print(documento)

# ================================= question-answering
while True:
    while True:
        print("Q&R | Faça uma pergunta sobre o texto: ")
        pergunta = input()
        if pergunta != None and len(pergunta) >= 4:
            break

    qa = pipeline("question-answering")
    question = pergunta
    result = qa(question=question, context=documento, max_length=100)
    print(f"\n\nResposta: {result['answer']}\n")
    print(f"\n\nQuer fazer outra pergunta(s/n) ou vamos para summarization")

    stay_qa = input()

    if stay_qa.lower() == "n":
        break

# ================================= summarization
resumo = pipeline("summarization")
result = resumo(documento, max_length=100, min_length=50,do_sample=False)
print(f"\n\nResumo: {result[0]['summary_text']}\n")

# ================================= sentiment-analysis

analise_sentimento = pipeline("sentiment-analysis")

p1 = """
A cafeteria dona Benta oferece bom atendimento e as bebidas estão sempre fresquinhas, 
a variedade dos cafés faz do lugar o único do tipo na cidade. No entanto, 
o tempo de espera tem se tornado um problema, espero que possam melhorar isto em breve."""

p2 = """
A cafeteria Dona Benta se destaca pelo excelente atendimento e pelas bebidas sempre fresquinhas. 
Com uma variedade única de cafés, é um verdadeiro diferencial na cidade. A popularidade do lugar 
cresce a cada dia, refletindo a preferência dos clientes. Tenho certeza de que a equipe continuará 
aprimorando a experiência para tornar cada visita ainda mais especial!
"""

p2_en = """
Dona Benta Café stands out for its excellent service and always fresh beverages. 
With a unique variety of coffees, it is truly one of a kind in the city. 
Its growing popularity reflects the customers preference, and I am sure the team will 
continue enhancing the experience to make every visit even more special!"""

samples_frase = [ p1, p2, p2_en]

for pgraphs in samples_frase:

    result = analise_sentimento(pgraphs)
    print("\n\nResultado:\n")
    for idx, sentimento in enumerate(result):
        print(f"\tFrase: {idx +1}: {sentimento}\n")


