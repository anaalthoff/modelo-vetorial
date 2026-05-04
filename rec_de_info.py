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