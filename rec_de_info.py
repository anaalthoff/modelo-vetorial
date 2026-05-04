# Problemas Clássicos da RI - Sinonímia, Polissemia e Ambiguidade

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Carregar modelo de embeddings (pré-treinado multilíngue)
print("Carregando modelo de embeddings...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# ============================================
# PARTE 1: SINONÍMIA
# ============================================
print("\n" + "="*80)
print("PARTE 1: PROBLEMA DA SINONÍMIA")
print("="*80)

consulta_sinonimia = "carro elétrico"

documentos_sinonimia = [
    "O automóvel elétrico é o futuro da mobilidade sustentável",  # sinônimo
    "Veículos movidos a eletricidade poluem menos",               # sinônimo
    "Carros elétricos têm autonomia cada vez maior",              # termo exato
    "A Tesla lidera o mercado de veículos elétricos",             # sinônimo
    "O preço das baterias para carros elétricos está caindo",     # termo exato
    "A gasolina ainda é o principal combustível para transporte"  # não relacionado
]

# Criar embeddings
emb_consulta_sin = model.encode([consulta_sinonimia])
emb_docs_sin = model.encode(documentos_sinonimia)

# Calcular similaridades
sim_sin = cosine_similarity(emb_consulta_sin, emb_docs_sin)[0]

print(f"\nConsulta: '{consulta_sinonimia}'")
print("\nResultados (ordenados por similaridade):")
indices_ordenados = np.argsort(sim_sin)[::-1]
for i, idx in enumerate(indices_ordenados):
    termo = "termo exato" if "carro" in documentos_sinonimia[idx].lower() else "sinônimo" if any(t in documentos_sinonimia[idx].lower() for t in ["automóvel", "veículo"]) else "não relacionado"
    print(f"{i+1}. Similaridade: {sim_sin[idx]:.4f} | {termo} | {documentos_sinonimia[idx][:60]}...")

# ============================================
# PARTE 2: POLISSEMIA
# ============================================
print("\n" + "="*80)
print("PARTE 2: PROBLEMA DA POLISSEMIA")
print("="*80)

consulta_polissemia = "manga"

documentos_polissemia = [
    # Sentido 1: Fruta
    "A manga é uma fruta tropical muito saborosa",
    "Receita de suco de manga com laranja",
    
    # Sentido 2: Vestuário
    "Como ajustar a manga de uma camisa social",
    "Camisas de manga curta são ideais para o verão",
    
    # Sentido 3: Anime (mangá)
    "O novo capítulo do mangá será lançado amanhã",
    "Onde comprar mangás em português"
]

sentidos = ["Fruta"] * 2 + ["Vestuário"] * 2 + ["Mangá"] * 2

# Criar embeddings
emb_consulta_pol = model.encode([consulta_polissemia])
emb_docs_pol = model.encode(documentos_polissemia)

# Calcular similaridades
sim_pol = cosine_similarity(emb_consulta_pol, emb_docs_pol)[0]

print(f"\nConsulta: '{consulta_polissemia}' (termo polissêmico)")
print("\nResultados por sentido:")

for sentido in set(sentidos):
    indices = [i for i, s in enumerate(sentidos) if s == sentido]
    print(f"\n{sentido}:")
    for i in indices:
        print(f"  Similaridade: {sim_pol[i]:.4f} | {documentos_polissemia[i]}")