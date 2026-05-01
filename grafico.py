# O código a seguir gera um gráfico 3D que plota esses vetores como setas a partir da origem, permitindo visualizar os ângulos entre eles.

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Configuração do Gráfico 3D ---
# Vetores no espaço 3D (usando as dimensões: gato, caça, rato)
# Consulta Q = (1, 1, 0)
# Documento D1 = (1, 1, 1)
# Documento D2 = (0, 1, 0)

# Ponto de origem para todos os vetores
origem = [0, 0, 0]

# Coordenadas dos vetores
Q = [1, 1, 0]
D1 = [1, 1, 1]
D2 = [0, 1, 0]

# Cria a figura e o eixo 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Função para desenhar um vetor colorido
def desenha_vetor(ax, vetor, cor, rotulo):
    ax.quiver(origem[0], origem[1], origem[2],
              vetor[0], vetor[1], vetor[2],
              color=cor, arrow_length_ratio=0.1, linewidth=2)
    # Adiciona um rótulo na ponta do vetor
    ax.text(vetor[0], vetor[1], vetor[2], rotulo, color=cor, fontsize=12, fontweight='bold')

# Desenha os vetores
desenha_vetor(ax, Q, 'red', 'Q (consulta)')
desenha_vetor(ax, D1, 'green', 'D1')
desenha_vetor(ax, D2, 'blue', 'D2')

# Configura os rótulos dos eixos
ax.set_xlabel('Dimensão: gato', fontsize=12)
ax.set_ylabel('Dimensão: caça', fontsize=12)
ax.set_zlabel('Dimensão: rato', fontsize=12)
ax.set_title('Visualização 3D da Similaridade por Cosseno', fontsize=14)

# Define os limites dos eixos para melhor visualização
ax.set_xlim(0, 1.5)
ax.set_ylim(0, 1.5)
ax.set_zlim(0, 1.5)

# Adiciona uma grade para melhor orientação espacial
ax.grid(True)

# Adiciona uma legenda descritiva no canto inferior direito da figura
# Como a legenda do quiver é complexa, adicionamos um texto explicativo
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
textstr = '\n'.join((
    r'Similaridades:',
    r'$\cos(Q, D1) = 0.82$',
    r'$\cos(Q, D2) = 0.41$'))
ax.text2D(0.85, 0.95, textstr, transform=ax.transAxes, fontsize=11,
          verticalalignment='top', bbox=props)

# Ajusta o ângulo de visualização para melhor perspectiva
ax.view_init(elev=20, azim=-45)

# Exibe o gráfico
plt.tight_layout()
plt.show()