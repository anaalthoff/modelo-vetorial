# Importação das bibliotecas
# pip install numpy scikit-learn pandas
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Para visualização mais amigável
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Documentos de exemplo
documentos = [
    "Python para ciência de dados",
    "Python para PLN",
    "Aprendizado de máquina para dados"
]

# Nomes dos documentos para referência
nomes_docs = ["Doc1", "Doc2", "Doc3"]

# Utiliza o CountVectorizer do scikit-learn para criar a matriz de contagens:

# Criar o vetorizador (considerando palavras individuais)
vectorizer = CountVectorizer()

# Ajustar o vetorizador aos documentos e transformar em matriz
X = vectorizer.fit_transform(documentos)

# Obter os nomes dos termos (vocabulário)
termos = vectorizer.get_feature_names_out()

# Converter para DataFrame para visualização
df_termos = pd.DataFrame(X.toarray(), columns=termos, index=nomes_docs)

print("Matriz Documento-Termo (frequências brutas):")
print(df_termos)
print("\n")

# Consulta do usuário
consulta = "ciência dados"

# Transformar a consulta usando o mesmo vetorizador
consulta_vetor = vectorizer.transform([consulta])

print("Vetor da consulta:")
print(pd.DataFrame(consulta_vetor.toarray(), columns=termos, index=["consulta"]))
print("\n")

# Calcular similaridade por cosseno entre a consulta e todos os documentos
similaridades = cosine_similarity(consulta_vetor, X)[0]

# Criar DataFrame com os resultados
resultados = pd.DataFrame({
    'Documento': nomes_docs,
    'Similaridade': similaridades,
    'Texto': documentos
}).sort_values('Similaridade', ascending=False)

print("Resultados ranqueados por similaridade:")
print(resultados)
print("\n")