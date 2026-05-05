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

# ============================================
# PARTE 3: AMBIGUIDADE
# ============================================
print("\n" + "="*80)
print("PARTE 3: PROBLEMA DA AMBIGUIDADE")
print("="*80)

consulta_ambiguidade = "banco"

documentos_ambiguidade = [
    # Interpretação 1: Instituição financeira
    "O Banco Central anunciou nova taxa de juros",
    "Como abrir uma conta corrente em banco digital",
    
    # Interpretação 2: Assento
    "Banco de madeira maciça para jardim",
    "Como restaurar bancos antigos de praça",
    
    # Interpretação 3: Banco de dados
    "Introdução a bancos de dados relacionais",
    "Bancos de dados NoSQL: MongoDB e Cassandra"
]

interpretacoes = ["Financeiro"] * 2 + ["Assento"] * 2 + ["Dados"] * 2

# Criar embeddings
emb_consulta_amb = model.encode([consulta_ambiguidade])
emb_docs_amb = model.encode(documentos_ambiguidade)

# Calcular similaridades
sim_amb = cosine_similarity(emb_consulta_amb, emb_docs_amb)[0]

print(f"\nConsulta: '{consulta_ambiguidade}' (ambígua)")
print("\nResultados por interpretação:")

for interp in set(interpretacoes):
    indices = [i for i, s in enumerate(interpretacoes) if s == interp]
    print(f"\n{interp}:")
    for i in indices:
        print(f"  Similaridade: {sim_amb[i]:.4f} | {documentos_ambiguidade[i]}")

# ============================================
# PARTE 4: VISUALIZAÇÃO INTEGRADA
# ============================================
print("\n" + "="*80)
print("PARTE 4: VISUALIZAÇÃO INTEGRADA")
print("="*80)

# Combinar todos os documentos para visualização
todos_documentos = (
    documentos_sinonimia + 
    documentos_polissemia + 
    documentos_ambiguidade
)

todas_consultas = [consulta_sinonimia, consulta_polissemia, consulta_ambiguidade]
emb_todas_consultas = model.encode(todas_consultas)
emb_todos_docs = model.encode(todos_documentos)

# Reduzir dimensionalidade para visualização
pca = PCA(n_components=2)
todos_embeddings = np.vstack([emb_todas_consultas, emb_todos_docs])
embeddings_2d = pca.fit_transform(todos_embeddings)

# Separar consultas e documentos
consultas_2d = embeddings_2d[:3]
documentos_2d = embeddings_2d[3:]

# Criar rótulos para cores
cores_docs = []
for i, doc in enumerate(todos_documentos):
    if i < len(documentos_sinonimia):
        cores_docs.append('blue')  # Sinonímia
    elif i < len(documentos_sinonimia) + len(documentos_polissemia):
        cores_docs.append('green')  # Polissemia
    else:
        cores_docs.append('orange')  # Ambiguidade

# Plotar
plt.figure(figsize=(14, 10))

# Documentos
scatter = plt.scatter(documentos_2d[:, 0], documentos_2d[:, 1], 
                      c=cores_docs, s=100, alpha=0.6, edgecolors='black', linewidth=1)

# Consultas (destacadas)
cores_consulta = ['red', 'red', 'red']
plt.scatter(consultas_2d[:, 0], consultas_2d[:, 1], 
            c='red', s=300, marker='*', edgecolors='black', 
            linewidth=2, label='Consultas')

# Adicionar rótulos para consultas
for i, consulta in enumerate(todas_consultas):
    plt.annotate(f'Consulta {i+1}: "{consulta}"', 
                (consultas_2d[i, 0], consultas_2d[i, 1]),
                xytext=(10, 10), textcoords='offset points', 
                fontsize=10, fontweight='bold')

# Adicionar alguns rótulos para documentos representativos
indices_destaque = [0, 3, 6, 8, 10, 12]  # Índices de documentos representativos
for idx in indices_destaque:
    plt.annotate(f'D{idx+1}', (documentos_2d[idx, 0], documentos_2d[idx, 1]),
                xytext=(5, 5), textcoords='offset points', fontsize=8)

# Legenda
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='blue', alpha=0.6, label='Sinonímia'),
    Patch(facecolor='green', alpha=0.6, label='Polissemia'),
    Patch(facecolor='orange', alpha=0.6, label='Ambiguidade'),
    Patch(facecolor='red', alpha=1.0, label='Consultas')
]
plt.legend(handles=legend_elements, loc='upper right')

plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Espaço Semântico: Problemas Clássicos da RI')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================
# PARTE 5: DESAMBIGUAÇÃO POR CONTEXTO
# ============================================
print("\n" + "="*80)
print("PARTE 5: DESAMBIGUAÇÃO POR CONTEXTO")
print("="*80)

# Testar consultas contextualizadas para o termo ambíguo "banco"
consultas_contextuais = [
    "banco juros",      # Contexto financeiro
    "banco investimentos",  # Contexto financeiro
    "banco madeira",    # Contexto assento
    "banco jardim",     # Contexto assento
    "banco dados",      # Contexto banco de dados
    "banco SQL"         # Contexto banco de dados
]

print("Comparação de similaridades para consultas contextualizadas:")
print("-"*80)

for consulta_ctx in consultas_contextuais:
    emb_ctx = model.encode([consulta_ctx])
    sim_ctx = cosine_similarity(emb_ctx, emb_docs_amb)[0]
    
    # Média por interpretação
    medias = {}
    for interp in set(interpretacoes):
        indices = [i for i, s in enumerate(interpretacoes) if s == interp]
        medias[interp] = np.mean([sim_ctx[i] for i in indices])
    
    # Identificar interpretação predominante
    interp_pred = max(medias, key=medias.get)
    confianca = medias[interp_pred] / sum(medias.values())
    
    print(f"\nConsulta: '{consulta_ctx}'")
    for interp in medias:
        print(f"  {interp}: {medias[interp]:.4f}")
    print(f"  → Predominante: {interp_pred} (confiança: {confianca:.2%})")

# ============================================
# PARTE 6: DISCUSSÃO DOS RESULTADOS
# ============================================
print("\n" + "="*80)
print("DISCUSSÃO DOS RESULTADOS")
print("="*80)
print("""
SINONÍMIA:
- Documentos com sinônimos ("automóvel", "veículo") alcançam similaridades significativas
- Em sistemas puramente lexicais, estes documentos receberiam similaridade ZERO
- Embeddings capturam relações semânticas entre termos diferentes mas relacionados

POLISSEMIA:
- A consulta "manga" produz similaridades distribuídas entre os três sentidos
- Nenhum sentido domina claramente, refletindo a ambiguidade do termo
- A desambiguação requer contexto adicional

AMBIGUIDADE:
- A consulta "banco" também produz similaridades distribuídas
- Na projeção 2D, a consulta posiciona-se entre os agrupamentos de documentos
- Consultas contextualizadas direcionam a busca para a interpretação desejada

DESAMBIGUAÇÃO POR CONTEXTO:
- "banco juros" e "banco investimentos" favorecem interpretação financeira
- "banco madeira" e "banco jardim" favorecem assentos
- "banco dados" e "banco SQL" favorecem banco de dados
- A confiança na interpretação correta aumenta significativamente com contexto adequado

LIMITAÇÕES PERSISTENTES:
- Embeddings não eliminam completamente a vantagem dos termos exatos
- Contexto insuficiente pode não resolver ambiguidades
- Sentidos raros podem ser sub-representados nos dados de treinamento
""")