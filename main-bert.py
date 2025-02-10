from transformers import pipeline, BertTokenizerFast, EncoderDecoderModel
import torch
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

    qa = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-wquad")
    question = pergunta
    result = qa(question=question, context=documento, max_length=100)

    print(f"\n\nResposta: {result['answer']}\n")
    print(f"\n\nQuer fazer outra pergunta(s/n) ou vamos para summarization")

    stay_qa = input()

    if stay_qa.lower() == "n":
        break

# ================================= summarization
model_bert = "mrm8488/bert-mini2bert-mini-finetuned-cnn_daily_mail-summarization"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizerFast.from_pretrained(model_bert)
model = EncoderDecoderModel.from_pretrained(model_bert)

def resumo():
    inputs = tokenizer({documento}, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask_to(device)

    output = model.generate(input_ids, attention_mask=attention_mask)

    return (f"\n\nResumo: {tokenizer.decode(output[0], skip_special_tokens=True)}\n")

resumo(documento)
# ================================= sentiment-analysis

model_sa = "nlptown/bert-base-multilingual-uncased-sentiment"

analise_sentimento = pipeline("sentiment-analysis", model=model_sa)

def mapear_sentimento(label):
    estrelas = int(label.split()[0])
    if estrelas in (1,2):
        return "Negativo"
    elif estrelas == 3:
        return "Neutro"
    else:
        return "Positivo"
    
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

samples_frase = [ p1, p2 ]

for pgraphs in samples_frase:
    resultados = analise_sentimento(pgraphs)
    for idx, resultado in enumerate(resultados):
        sentimento = mapear_sentimento(resultado['label'])
        print(f"\n\nTexto: {idx +1}: {documento}")
        print(f"Classificação: {sentimento} (Confiança: {resultado['score']:2f})\n")
