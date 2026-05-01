# Importa as bibliotecas necessárias
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Corpus de documentos e consulta
documentos = [
    "O gato caça o rato",  # Documento 1
    "O cão caça a bola"     # Documento 2
]
consulta = ["gato caça"] # A consulta do usuário

# 2. Vetorização usando CountVectorizer (converte texto em vetores de contagem - TF)
#    O parâmetro binary=True garante que a contagem seja 0 ou 1, simplificando o exemplo.
#    Em um cenário real, usaríamos TfidfVectorizer para aplicar TF-IDF.
vectorizer = CountVectorizer(binary=True, lowercase=True)

# Ajusta o vetorizador ao corpus e transforma os documentos em vetores
# É crucial transformar a consulta *depois* de ajustar ao corpus, para garantir
# que o vocabulário (e a ordem das dimensões) seja o mesmo.
X_documentos = vectorizer.fit_transform(documentos)
X_consulta = vectorizer.transform(consulta)

# 3. Exibe os vetores criados
print("Vocabulário (termos):", vectorizer.get_feature_names_out())
print("\nVetor do Documento 1 (D1):", X_documentos[0].toarray())
print("Vetor do Documento 2 (D2):", X_documentos[1].toarray())
print("Vetor da Consulta (Q):    ", X_consulta.toarray())

# 4. Cálculo da Similaridade por Cosseno
#    A função cosine_similarity calcula a similaridade entre todos os pares de vetores fornecidos.
#    Passamos a consulta e os documentos. O resultado é uma matriz 1x2.
similaridades = cosine_similarity(X_consulta, X_documentos)

print("\n--- Cálculo da Similaridade por Cosseno ---")
print(f"Similaridade entre Consulta e Documento 1: {similaridades[0][0]:.3f}")
print(f"Similaridade entre Consulta e Documento 2: {similaridades[0][1]:.3f}")

# 5. Ranqueamento dos resultados
#    Obtemos os índices dos documentos ordenados por similaridade (decrescente)
indices_ordenados = np.argsort(similaridades[0])[::-1]

print("\n--- Resultados Ranqueados ---")
print(f"Consulta do usuário: {consulta}")
for i, idx in enumerate(indices_ordenados):
    print(f"{i+1}º - Documento {idx+1} (Score: {similaridades[0][idx]:.3f}): '{documentos[idx]}'")